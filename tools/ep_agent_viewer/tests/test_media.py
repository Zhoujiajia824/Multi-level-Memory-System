"""Exercise browser-compatible copies without touching original video bytes."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from epviz.media import MediaCache


class MediaTests(unittest.TestCase):
    def test_missing_encoder_and_cache_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "original.mp4"
            source.write_bytes(b"original")
            with patch("epviz.media.shutil.which", return_value=None):
                cache = MediaCache(Path(folder) / "cache")
            self.assertEqual(cache.status(source)["status"], "unavailable")
            for key in ("../original.mp4", "secret", "a" * 64 + "/../original.mp4"):
                with self.assertRaises(ValueError):
                    cache.resolve(key)
            previous = cache.key(source)
            source.write_bytes(b"new file")
            self.assertNotEqual(cache.key(source), previous)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is optional")
    def test_h264_copy_decodes_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "original.mp4"
            subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "lavfi", "-i", "color=c=black:s=32x32:r=10", "-t", "0.3",
                            "-c:v", "mpeg4", str(source)], check=True, capture_output=True, timeout=15)
            original = hashlib.sha256(source.read_bytes()).digest()
            cache = MediaCache(Path(folder) / "cache")
            state = cache.status(source, start=True)
            deadline = time.monotonic() + 20
            while state["status"] == "working" and time.monotonic() < deadline:
                time.sleep(.05)
                state = cache.status(source)
            self.assertEqual(state["status"], "ready", state)
            target = cache.resolve(state["url"].rsplit("/", 1)[-1])
            self.assertNotEqual(target, source)
            self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), original)
            decode = subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error",
                                     "-i", str(target), "-f", "null", "-"], capture_output=True, timeout=15)
            self.assertEqual(decode.returncode, 0, decode.stderr)


if __name__ == "__main__":
    unittest.main()
