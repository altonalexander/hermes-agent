"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.
"""

import base64
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import (  # noqa: E402
    _StreamingMediaResolver,
    _resolve_media_to_data_urls,
)

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


class TestResolveMediaToDataUrls(unittest.TestCase):
    def _write_png(self, tmpdir_name="hermes_media_test"):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix=tmpdir_name))
        p = d / "shot.png"
        p.write_bytes(_PNG_BYTES)
        return p

    def test_media_tag_inlined(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"Here you go: MEDIA:{p}")
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)

    def test_backtick_wrapped_tag(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"See `MEDIA:{p}` above")
        self.assertIn("data:image/png;base64,", out)

    def test_missing_file_left_untouched(self):
        text = "MEDIA:/nonexistent/path/shot.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_non_image_left_untouched(self):
        text = "MEDIA:/tmp/archive.zip"
        self.assertEqual(_resolve_media_to_data_urls(text), text)


class TestStreamingMediaResolver(unittest.TestCase):
    """Streamed replies never reach ``final_response`` resolution, so the
    resolver has to inline tags as the deltas go out."""

    def _stream(self, deltas):
        r = _StreamingMediaResolver()
        return [r.feed(d) for d in deltas] + [r.flush()]

    def test_tag_split_across_deltas_is_inlined(self):
        p = TestResolveMediaToDataUrls()._write_png()
        text = f"Here it is:\n\nMEDIA:{p}"
        cut = text.index("MEDIA:") + 3
        out = self._stream([text[:cut - 5], text[cut - 5:cut], text[cut:cut + 4], text[cut + 4:]])
        joined = "".join(out)
        self.assertIn("data:image/png;base64,", joined)
        self.assertNotIn("MEDIA:", joined)
        self.assertTrue(joined.startswith("Here it is:"))

    def test_plain_text_passes_through_immediately(self):
        r = _StreamingMediaResolver()
        self.assertEqual(r.feed("hello "), "hello ")
        self.assertEqual(r.feed("world"), "world")
        self.assertEqual(r.flush(), "")

    def test_text_after_tag_line_is_released(self):
        p = TestResolveMediaToDataUrls()._write_png()
        r = _StreamingMediaResolver()
        self.assertEqual(r.feed(f"MEDIA:{p}"), "")
        out = r.feed("\nDone.")
        self.assertIn("data:image/png;base64,", out)
        self.assertTrue(out.endswith("Done."))

    def test_unresolvable_tag_is_left_as_text(self):
        out = "".join(self._stream(["MEDIA:/nonexistent/", "shot.png"]))
        self.assertEqual(out, "MEDIA:/nonexistent/shot.png")


if __name__ == "__main__":
    unittest.main()
