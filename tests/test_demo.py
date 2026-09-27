"""Local demo contract tests; never open the independent evaluation set."""
import base64
import io
import unittest

from PIL import Image

from demo.app import RequestError, decode_image, warning_for_task
from demo.inference import CLASSES, build_question


class DemoTests(unittest.TestCase):
    def test_ui_vocabulary_and_canonical_training_prompts(self):
        self.assertEqual(len(CLASSES), 20)
        self.assertEqual(len({item["value"] for item in CLASSES}), 20)
        self.assertEqual(build_question("existence", "cat", None), "图中是否有猫？")
        self.assertEqual(build_question("counting", "cat", None), "图中一共有几个猫？")
        self.assertEqual(build_question("attribute", None, None), "图片主要是什么？")
        self.assertEqual(build_question("listing", None, None), "图中有什么物体？")
        self.assertEqual(build_question("spatial", "cat", "dog"), "猫在狗的左边还是右边？")

    def test_question_controls_reject_unsupported_requests(self):
        for args in (("counting", "dragon", None), ("spatial", "cat", "cat"),
                     ("spatial", "cat", None), ("color", "cat", None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                build_question(*args)

    def test_upload_is_decoded_in_memory_and_mime_checked(self):
        image = Image.new("RGB", (48, 40), (255, 0, 0))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        payload = base64.b64encode(encoded.getvalue()).decode("ascii")
        decoded = decode_image("data:image/png;base64," + payload)
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (48, 40))
        decoded.close()
        with self.assertRaises(RequestError):
            decode_image("data:image/jpeg;base64," + payload)
        with self.assertRaises(RequestError):
            decode_image("data:text/plain;base64," + payload)

    def test_task_specific_warnings(self):
        self.assertIn("多个同类物体", warning_for_task("counting"))
        self.assertIn("不支持可靠判断", warning_for_task("spatial"))
        self.assertIn("颜色", warning_for_task("attribute"))


if __name__ == "__main__":
    unittest.main()
