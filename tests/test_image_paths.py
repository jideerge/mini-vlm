import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets.image_paths import resolve_image_path


class ImagePathTests(unittest.TestCase):
    def test_original_file_is_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "image.jpg"
            p.write_bytes(b"x")
            self.assertEqual(resolve_image_path(str(p)), p)

    def test_windows_absolute_path_uses_exported_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported = Path(tmp) / "000001.jpg"
            exported.write_bytes(b"x")
            self.assertEqual(resolve_image_path(r"D:\VOC2007\JPEGImages\000001.jpg", tmp), exported)

    def test_environment_variable_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported = Path(tmp) / "000002.jpg"
            exported.write_bytes(b"x")
            with patch.dict("os.environ", {"MINIVLM_IMAGE_ROOT": tmp}):
                self.assertEqual(resolve_image_path(r"D:\VOC2007\JPEGImages\000002.jpg"), exported)

    def test_missing_image_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "000003.jpg"):
                resolve_image_path(r"D:\VOC2007\JPEGImages\000003.jpg", tmp)


if __name__ == "__main__":
    unittest.main()
