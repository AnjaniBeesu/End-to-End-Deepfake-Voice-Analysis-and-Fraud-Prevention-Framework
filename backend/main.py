from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from schemas import AnalysisResponse, HealthResponse
from services.audio_processor import validate_and_load_audio
from services.feature_extractor import extract_acoustic_features
from services.detector import detect_synthetic_voice

app = FastAPI(
    title="Deepfake Audio Detection API",
    description="End-to-end framework for acoustic feature extraction, synthetic voice analysis, and fraud prevention.",
    version="1.0.0",
)

# Enable CORS for Next.js frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to the Deepfake Audio Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    return HealthResponse(
        status="healthy",
        service="deepfake-voice-analysis-engine",
        version="1.0.0",
        audio_backend="scipy-numpy-dsp",
    )


@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze audio file for synthetic deepfake voice signatures",
)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Ingests an audio file (.wav, .mp3, .flac, etc.), extracts multi-dimensional
    acoustic and spectral characteristics, and executes the deepfake fraud detection engine.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}"
        )

    # 1. Validate and decode audio into normalized float32 mono stream
    audio, sample_rate = validate_and_load_audio(file_bytes, file.filename)

    # 2. Extract acoustic and spectral features
    features = extract_acoustic_features(audio, sample_rate)

    # 3. Detect synthetic voice probability and fraud indicators
    (
        fake_probability,
        confidence_score,
        risk_level,
        is_synthetic,
        summary,
        indicators,
        breakdown,
    ) = detect_synthetic_voice(features, audio, sample_rate)

    timestamp = datetime.now(timezone.utc).isoformat()

    return AnalysisResponse(
        filename=file.filename,
        status="completed",
        fake_probability=fake_probability,
        confidence_score=confidence_score,
        risk_level=risk_level,
        is_synthetic=is_synthetic,
        summary=summary,
        indicators=indicators,
        features=features,
        breakdown=breakdown,
        timestamp=timestamp,
    )
