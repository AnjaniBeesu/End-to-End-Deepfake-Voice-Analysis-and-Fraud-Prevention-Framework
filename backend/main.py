from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="Deepfake Audio Detection API")

SUPPORTED_EXTENSIONS = {".wav", ".wave"}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Audio Detection API"}


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

    # TODO: ML Contributors will integrate the PyTorch inference here
    return {
        "filename": filename,
        "status": "processing",
        "fake_probability": 0.0,
    }
