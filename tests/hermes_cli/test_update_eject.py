"""Tests for ``hermes update --eject`` (hermes_cli/update_cmd.py::cmd_update_eject).

Uses real git repos (not mocks) per the repo's E2E-validation preference.
A bundled checkout is a PLAIN SOURCE TREE without ``.git`` (the desktop
payload ships no repository), so the fixtures build one from a local
"origin" via ``git archive``, and eject grafts a fresh repository onto it
by fetching from that origin.
"""

import subprocess

import pytest

import hermes_cli.update_cmd as update_cmd
from hermes_cli.install_manifest import (
    CHANNEL_MAIN,
    CHANNEL_STABLE,
    MODE_BUNDLED,
    MODE_SOURCE,
    STYLE_EJECTED,
    is_ejected,
    read_install_manifest,
    write_install_manifest,
)
from hermes_cli.update_cmd import cmd_update_eject


def _git(cwd, *args):
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def origin(tmp_path):
    """A local origin with three commits and a v0.1.0 tag on main."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "test")
    # The real repository ignores the manifest and stamp in the checkout
    # root; the fixture mirrors that so post-eject `git status` contracts
    # match production.
    (origin / ".gitignore").write_text("/.hermes-install.json\n/.hermes_build_info.json\n")
    for i in range(3):
        (origin / f"f{i}.txt").write_text(f"content {i}\n")
        _git(origin, "add", ".")
        _git(origin, "commit", "-m", f"commit {i}")
    _git(origin, "tag", "v0.1.0")
    return origin


@pytest.fixture
def bundled_checkout(tmp_path, origin, monkeypatch):
    """A gitless source tree at v0.1.0, manifest-marked as bundled.

    This is exactly the shape payload_stage_repo materializes: the tag's
    tracked files, no .git. Eject fetches from OFFICIAL_REPO_URL, which the
    fixture points at the local origin.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    archive = tmp_path / "repo.tar"
    _git(origin, "archive", "--format=tar", "-o", str(archive), "v0.1.0")
    subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(checkout)],
        capture_output=True, check=True,
    )
    write_install_manifest(
        {"installMode": MODE_BUNDLED, "channel": CHANNEL_STABLE, "pinnedTag": "v0.1.0"},
        checkout,
    )
    monkeypatch.setattr(update_cmd, "OFFICIAL_REPO_URL", f"file://{origin}")
    return checkout


class _Args:
    def __init__(self, channel=None):
        self.eject = True
        self.channel = channel


def _patch_project_root(monkeypatch, root):
    import hermes_cli.main as hermes_main

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", root)


class TestEjectBundled:
    def test_eject_grafts_git_and_flips_mode(self, bundled_checkout, monkeypatch, capsys):
        _patch_project_root(monkeypatch, bundled_checkout)
        assert not (bundled_checkout / ".git").exists()

        rc = cmd_update_eject(_Args())

        assert rc == 0
        # A real repository now exists, at the pinned tag, on main.
        assert (bundled_checkout / ".git").is_dir()
        assert _git(bundled_checkout, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        assert _git(bundled_checkout, "rev-parse", "HEAD") == _git(
            bundled_checkout, "rev-parse", "tags/v0.1.0^{commit}"
        )
        # The graft changed no tracked file: the tree already matched the tag.
        assert _git(bundled_checkout, "status", "--porcelain") == ""
        manifest = read_install_manifest(bundled_checkout)
        assert manifest["installMode"] == MODE_SOURCE
        assert manifest["channel"] == CHANNEL_MAIN  # default eject channel
        # Sticky opt-out for silent auto-adoption.
        assert manifest["manageStyle"] == STYLE_EJECTED
        assert is_ejected(bundled_checkout)
        # Provenance pin survives for forensics.
        assert manifest["pinnedTag"] == "v0.1.0"
        assert "Ejected" in capsys.readouterr().out

    def test_eject_with_stable_channel(self, bundled_checkout, monkeypatch):
        _patch_project_root(monkeypatch, bundled_checkout)
        rc = cmd_update_eject(_Args(channel="stable"))
        assert rc == 0
        assert read_install_manifest(bundled_checkout)["channel"] == CHANNEL_STABLE

    def test_failed_fetch_leaves_manifest_and_tree_untouched(
        self, bundled_checkout, monkeypatch, capsys
    ):
        """Eject must be atomic-ish: no mode flip, no leftover .git on failure."""
        _patch_project_root(monkeypatch, bundled_checkout)
        monkeypatch.setattr(update_cmd, "OFFICIAL_REPO_URL", "file:///nonexistent/repo")

        rc = cmd_update_eject(_Args())

        assert rc == 1
        manifest = read_install_manifest(bundled_checkout)
        assert manifest["installMode"] == MODE_BUNDLED
        # A half-initialized repository must not survive: the next eject
        # retry starts clean, and bundled mode treats .git as foreign.
        assert not (bundled_checkout / ".git").exists()
        assert "aborted" in capsys.readouterr().out

    def test_missing_pinned_tag_refuses(self, tmp_path, monkeypatch, capsys):
        root = tmp_path / "notag"
        root.mkdir()
        write_install_manifest(
            {"installMode": MODE_BUNDLED, "channel": CHANNEL_STABLE}, root
        )
        _patch_project_root(monkeypatch, root)

        rc = cmd_update_eject(_Args())

        assert rc == 1
        assert read_install_manifest(root)["installMode"] == MODE_BUNDLED
        assert "no valid pinned tag" in capsys.readouterr().out

    def test_eject_with_preexisting_git_keeps_it(self, bundled_checkout, origin, monkeypatch):
        """A checkout that somehow already has .git (legacy adopted install)
        ejects through the same path without re-initializing."""
        _patch_project_root(monkeypatch, bundled_checkout)
        _git(bundled_checkout, "init", "-b", "main")
        _git(bundled_checkout, "remote", "add", "origin", f"file://{origin}")

        rc = cmd_update_eject(_Args())

        assert rc == 0
        manifest = read_install_manifest(bundled_checkout)
        assert manifest["installMode"] == MODE_SOURCE
        assert _git(bundled_checkout, "rev-parse", "HEAD") == _git(
            bundled_checkout, "rev-parse", "tags/v0.1.0^{commit}"
        )


class TestEjectOnSourceInstalls:
    def test_noop_without_channel(self, tmp_path, monkeypatch, capsys):
        root = tmp_path / "src"
        root.mkdir()
        _patch_project_root(monkeypatch, root)

        rc = cmd_update_eject(_Args())

        assert rc == 0
        assert "already source-managed" in capsys.readouterr().out
        # No manifest is created by a pure no-op eject... actually one may be
        # absent entirely; reading it must still say source/main.
        manifest = read_install_manifest(root)
        assert manifest["installMode"] == MODE_SOURCE

    def test_channel_switch_shorthand(self, tmp_path, monkeypatch):
        """--eject --channel stable on a source install just sets the channel."""
        root = tmp_path / "src"
        root.mkdir()
        _patch_project_root(monkeypatch, root)

        rc = cmd_update_eject(_Args(channel="stable"))

        assert rc == 0
        manifest = read_install_manifest(root)
        assert manifest["installMode"] == MODE_SOURCE
        assert manifest["channel"] == CHANNEL_STABLE
        # A channel switch on a never-desktop-managed checkout is NOT an
        # adoption opt-out — the checkout stays adoptable.
        assert not is_ejected(root)
