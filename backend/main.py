from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from starlette.responses import JSONResponse

app = FastAPI(title="Deepfake Audio Detection API")

SUPPORTED_EXTENSIONS = {".wav", ".wave"}
MAX_UPLOAD_SIZE = 25 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


@app.middleware("http")
async def limit_upload_body(request: Request, call_next):
    """Reject an oversized request while its ASGI body is being received."""
    if request.method != "POST" or request.url.path != "/api/v1/analyze":
        return await call_next(request)

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Upload is too large. Maximum request size is 25 MB."},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header."},
            )

    received = 0

    async def limited_receive():
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > MAX_UPLOAD_SIZE:
                raise UploadTooLarge
        return message

    request = Request(request.scope, receive=limited_receive)
    try:
        return await call_next(request)
    except UploadTooLarge:
        return JSONResponse(
            status_code=413,
            content={"detail": "Upload is too large. Maximum request size is 25 MB."},
        )


class UploadTooLarge(Exception):
    """Raised when the streamed request body exceeds the configured limit."""


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

    # The middleware caps the incoming multipart request as it streams in.
    # Keep file reads bounded; do not load the full upload into memory.
    total_size = 0
    while chunk := await file.read(UPLOAD_CHUNK_SIZE):
        total_size += len(chunk)

    await file.seek(0)

    # TODO: ML Contributors will integrate the PyTorch inference here
    return {
        "filename": filename,
        "status": "processing",
        "fake_probability": 0.0,
    }
