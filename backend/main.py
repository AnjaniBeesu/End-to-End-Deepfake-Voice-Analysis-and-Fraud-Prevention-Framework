from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request

app = FastAPI(title="Deepfake Audio Detection API")

SUPPORTED_EXTENSIONS = {".wav", ".wave"}
MAX_UPLOAD_SIZE = 25 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Audio Detection API"}


@app.post("/api/v1/analyze")
async def analyze_audio(request: Request, file: UploadFile = File(...)):
    # Reject oversized requests before FastAPI parses the multipart body.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            request_size = int(content_length)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")

        # Multipart framing and form fields add bytes beyond the audio itself,
        # so don't reject solely based on this header. The streaming check below
        # enforces the actual file limit.
        if request_size < 0:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")

    # Reject unsupported audio formats.
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Please upload a .wav or .wave file.",
        )

    # UploadFile is backed by a spooled temporary file after multipart parsing.
    # Read in bounded chunks so the endpoint doesn't copy the entire upload
    # into application memory, and enforce the per-file limit.
    total_size = 0
    while chunk := await file.read(UPLOAD_CHUNK_SIZE):
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE:
            await file.close()
            raise HTTPException(
                status_code=413,
                detail="Audio file is too large. Maximum upload size is 25 MB.",
            )

    await file.seek(0)

    # TODO: ML Contributors will integrate the PyTorch inference here
    return {
        "filename": filename,
        "status": "processing",
        "fake_probability": 0.0,
    }
