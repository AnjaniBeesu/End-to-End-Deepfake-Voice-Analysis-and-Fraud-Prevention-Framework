import io
import numpy as np
from scipy.io import wavfile
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _generate_test_wav(
    duration: float = 1.0,
    sample_rate: int = 16000,
    freq: float = 440.0,
    is_stereo: bool = False
) -> bytes:
    """Helper to generate an in-memory WAV file for testing."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    # Add fundamental + harmonics for realistic signal
    signal = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    audio = (signal * 16384).astype(np.int16)

    if is_stereo:
        audio = np.column_stack((audio, audio))

    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio)
    return buf.getvalue()


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["version"] == "1.0.0"


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "deepfake-voice-analysis-engine"


def test_analyze_valid_audio():
    wav_bytes = _generate_test_wav(duration=1.2, sample_rate=16000, freq=300.0)
    files = {"file": ("sample_voice.wav", wav_bytes, "audio/wav")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["filename"] == "sample_voice.wav"
    assert data["status"] == "completed"
    assert 0.0 <= data["fake_probability"] <= 1.0
    assert 0.0 <= data["confidence_score"] <= 100.0
    assert data["risk_level"] in ["LOW_RISK", "SUSPICIOUS", "HIGH_RISK_DEEPFAKE"]
    assert isinstance(data["indicators"], list)

    features = data["features"]
    assert features["sample_rate"] == 16000
    assert features["duration_seconds"] == pytest.approx(1.2, abs=0.05)
    assert features["channels"] == 1
    assert features["zero_crossing_rate"] >= 0.0
    assert features["spectral_centroid_hz"] > 0.0

    breakdown = data["breakdown"]
    assert 0.0 <= breakdown["spectral_consistency"] <= 100.0
    assert 0.0 <= breakdown["pitch_naturalness"] <= 100.0
    assert 0.0 <= breakdown["vocoder_artifact_score"] <= 100.0


def test_analyze_stereo_audio():
    wav_bytes = _generate_test_wav(duration=1.0, is_stereo=True)
    files = {"file": ("stereo_sample.wav", wav_bytes, "audio/wav")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["features"]["channels"] == 1


def test_analyze_empty_file():
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_unsupported_format():
    files = {"file": ("document.txt", b"plain text content", "text/plain")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_analyze_too_short_audio():
    # 0.1s is below MIN_DURATION_SECONDS (0.25s)
    short_wav = _generate_test_wav(duration=0.1)
    files = {"file": ("short.wav", short_wav, "audio/wav")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower()
