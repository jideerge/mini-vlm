"""Local-only upload demo for the frozen Mini-VLM B/16 baseline.

Uses Python's standard-library HTTP server so the existing project venv needs
no web-framework download. Uploaded images stay in memory and are never saved.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import re
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.inference import CLASSES, LocalMiniVLM, build_question  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
IMAGE_MIME = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}


class RequestError(ValueError):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def decode_image(data_url: object) -> Image.Image:
    """Validate and decode a browser data URL without writing an upload file."""
    if not isinstance(data_url, str):
        raise RequestError("请先选择 JPEG、PNG 或 WebP 图片。")
    match = re.fullmatch(r"data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/=]+)", data_url)
    if match is None:
        raise RequestError("只支持 JPEG、PNG 或 WebP 图片。")
    encoded = match.group(2)
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise RequestError("图片不能超过 8 MB。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise RequestError("图片编码损坏，请重新选择文件。") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise RequestError("图片不能为空或超过 8 MB。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            if probe.format != IMAGE_MIME[match.group(1)]:
                raise RequestError("图片内容与文件格式不符。")
            width, height = probe.size
            if width < 32 or height < 32 or width * height > MAX_IMAGE_PIXELS:
                raise RequestError("图片尺寸应至少为 32×32，且不超过 2000 万像素。")
            probe.verify()
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            image.load()
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RequestError("无法读取这张图片，请换一张 JPEG、PNG 或 WebP 图片。") from exc


def warning_for_task(task: str) -> str:
    if task == "spatial":
        return "空间位置暂不支持可靠判断；下方只是实验性输出，不应据此作判断。"
    if task == "counting":
        return "多个同类物体的计数尤其容易出错，请勿把结果当作可靠数量。"
    if task == "attribute":
        return "训练时的“属性”任务是回答画面主要物体的 VOC 类别，并不判断颜色或材质。"
    return "实验性结果，请自行核对图像。"


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], device: str):
        super().__init__(address, DemoHandler)
        self.device = device
        self.model: LocalMiniVLM | None = None
        self.model_lock = threading.Lock()


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, value: dict) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _local_host(self) -> bool:
        hostname = self.headers.get("Host", "").split(":", 1)[0].lower()
        return hostname in ("127.0.0.1", "localhost")

    def do_GET(self) -> None:  # noqa: N802
        if not self._local_host():
            self._json(HTTPStatus.FORBIDDEN, {"error": "仅允许本机访问。"})
            return
        if self.path == "/api/config":
            self._json(HTTPStatus.OK, {"classes": CLASSES})
            return
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok", "model_loaded": self.server.model is not None})
            return
        asset = STATIC.get(self.path)
        if asset is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "页面不存在。"})
            return
        filename, content_type = asset
        self._send(HTTPStatus.OK, (DEMO_DIR / filename).read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_host():
            self._json(HTTPStatus.FORBIDDEN, {"error": "仅允许本机访问。"})
            return
        if self.path != "/api/predict":
            self._json(HTTPStatus.NOT_FOUND, {"error": "接口不存在。"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请求格式必须为 JSON。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise RequestError("请求为空或图片超过 8 MB。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise RequestError("请求内容无效。")
            task = payload.get("task")
            question = build_question(task, payload.get("category"), payload.get("other_category"))
            image = decode_image(payload.get("image_data"))
            try:
                # The model is loaded once. Serial GPU access avoids overlapping
                # requests exhausting the local 6 GB card.
                with self.server.model_lock:
                    if self.server.model is None:
                        self.server.model = LocalMiniVLM(device=self.server.device)
                    answer = self.server.model.predict(image, question)
            finally:
                image.close()
            self._json(HTTPStatus.OK, {
                "question": question,
                "answer": answer.strip() or "模型未生成可展示的答案。",
                "warning": warning_for_task(task),
            })
        except RequestError as exc:
            self._json(exc.status, {"error": str(exc)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": "本地推理失败。请检查启动终端中的错误信息。"
            })

    def log_message(self, format: str, *args: object) -> None:
        print("[demo] " + format % args, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mini-VLM local-only demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    server = DemoServer(("127.0.0.1", args.port), args.device)
    print(f"Mini-VLM 本地 demo：http://127.0.0.1:{args.port}", flush=True)
    print("首次提问会加载模型；按 Ctrl+C 停止。图片仅在内存中处理。", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
