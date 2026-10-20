from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.responses import JSONResponse

app = FastAPI(title="Deepfake Audio Detection API")

SUPPORTED_EXTENSIONS = {".wav", ".wave"}
MAX_UPLOAD_SIZE = 25 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
UPLOAD_TOO_LARGE_DETAIL = "Upload is too large. Maximum request size is 25 MB."


class UploadSizeLimitMiddleware:
    """Limit the streamed request body for the audio-analysis endpoint."""

    def __init__(self, app, max_bytes: int = MAX_UPLOAD_SIZE):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/api/v1/analyze"
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._respond(send, 400, "Invalid Content-Length header.")
                return
            if declared_length < 0:
                await self._respond(send, 400, "Invalid Content-Length header.")
                return
            if declared_length > self.max_bytes:
                await self._respond(send, 413, UPLOAD_TOO_LARGE_DETAIL)
                return

        received = 0
        too_large = False

        async def limited_receive():
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    too_large = True
                    raise _UploadTooLarge
            return message

        async def guarded_send(message):
            # Starlette may try to emit an error response after limited_receive
            # raises. The middleware owns the 413 response in that case.
            if not too_large:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _UploadTooLarge:
            await self._respond(send, 413, UPLOAD_TOO_LARGE_DETAIL)

    @staticmethod
    async def _respond(send, status_code: int, detail: str):
        response = JSONResponse(status_code=status_code, content={"detail": detail})
        await response(
            {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "method": "POST", "path": "/api/v1/analyze", "raw_path": b"/api/v1/analyze",
             "query_string": b"", "headers": [], "client": None, "server": None,
             "scheme": "http", "http_version": "1.1"},
            _empty_receive,
            send,
        )


async def _empty_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


class _UploadTooLarge(Exception):
    """Internal signal used to stop reading an oversized request."""


app.add_middleware(UploadSizeLimitMiddleware)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Audio Detection API"}


@app.post("/api/v1/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Please upload a .wav or .wave file.",
        )

    # The middleware enforces the request limit while bytes are arriving.
    # Read in bounded chunks so the endpoint never loads the full file at once.
    while await file.read(UPLOAD_CHUNK_SIZE):
        pass

    # TODO: ML Contributors will integrate the PyTorch inference here.
    return {
        "filename": filename,
        "status": "processing",
        "fake_probability": 0.0,
    }
