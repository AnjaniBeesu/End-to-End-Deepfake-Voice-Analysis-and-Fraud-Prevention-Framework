from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW_RISK = "LOW_RISK"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK_DEEPFAKE = "HIGH_RISK_DEEPFAKE"


class AcousticFeatures(BaseModel):
    sample_rate: int = Field(..., description="Audio sampling rate in Hz")
    duration_seconds: float = Field(..., description="Total duration of audio in seconds")
    channels: int = Field(..., description="Number of audio channels (normalized to 1)")
    zero_crossing_rate: float = Field(..., description="Normalized zero crossing rate")
    rms_energy: float = Field(..., description="Root Mean Square energy of the signal")
    spectral_centroid_hz: float = Field(..., description="Spectral centroid indicating brightness of sound")
    spectral_rolloff_hz: float = Field(..., description="Frequency below which 85% of spectral energy lies")
    spectral_flatness: float = Field(..., description="Measure of noisiness vs tonality in spectrum")
    fundamental_frequency_hz: Optional[float] = Field(None, description="Estimated average pitch (F0) in Hz")
    pitch_jitter: Optional[float] = Field(None, description="Cycle-to-cycle fundamental frequency variation")
    shimmer: Optional[float] = Field(None, description="Cycle-to-cycle amplitude perturbation")


class AnalysisBreakdown(BaseModel):
    spectral_consistency: float = Field(..., description="Consistency of harmonic spectral distribution (0-100)")
    pitch_naturalness: float = Field(..., description="Human-like pitch modulation score (0-100)")
    vocal_tract_dynamics: float = Field(..., description="Realistic vocal tract filter dynamics (0-100)")
    vocoder_artifact_score: float = Field(..., description="Probability of neural vocoder artifacts (0-100)")


class AnalysisResponse(BaseModel):
    filename: str
    status: str = "completed"
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Synthetic voice probability from 0.0 to 1.0")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Detection confidence percentage")
    risk_level: RiskLevel
    is_synthetic: bool
    summary: str
    indicators: List[str] = Field(default_factory=list)
    features: AcousticFeatures
    breakdown: AnalysisBreakdown
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    audio_backend: str
