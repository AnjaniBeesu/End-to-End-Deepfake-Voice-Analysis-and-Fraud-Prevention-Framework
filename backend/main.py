from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="Deepfake Audio Detection API")

SUPPORTED_EXTENSIONS = {".wav", ".wave"}
MAX_UPLOAD_SIZE = 25 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Audio Detection API"}


async def validate_upload_size(file: UploadFile) -> int:
    """Count uploaded bytes in bounded chunks and reject files over the limit."""
    total_size = 0

    while chunk := await file.read(UPLOAD_CHUNK_SIZE):
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Audio file is too large. Maximum upload size is 25 MB.",
            )

    # Rewind so downstream processing can read the validated upload.
    await file.seek(0)
    return total_size


@app.post("/api/v1/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    # Reject unsupported audio formats.
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Please upload a .wav or .wave file.",
        )

    await validate_upload_size(file)

    # TODO: ML Contributors will integrate the PyTorch inference here
    return {
        "filename": filename,
        "status": "processing",
        "fake_probability": 0.0,
    }
