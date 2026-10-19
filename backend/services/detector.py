import numpy as np
from schemas import AcousticFeatures, AnalysisBreakdown, RiskLevel


def detect_synthetic_voice(
    features: AcousticFeatures,
    audio: np.ndarray,
    sample_rate: int
) -> tuple[float, float, RiskLevel, bool, str, list[str], AnalysisBreakdown]:
    """
    Evaluates acoustic and spectral features to determine the probability that the
    audio sample is a synthetic deepfake voice.

    Returns:
        (fake_probability, confidence_score, risk_level, is_synthetic, summary, indicators, breakdown)
    """
    indicators: list[str] = []
    synthetic_votes = []
    weights = []

    # 1. Pitch Naturalness & Micro-Perturbation Analysis
    # Human vocal cords have natural micro-instability (organic jitter ~0.004 to 0.030).
    pitch_naturalness = 85.0
    if features.pitch_jitter is not None:
        jitter = features.pitch_jitter
        if jitter < 0.003:
            # Unnaturally monotonic / robotic pitch
            synthetic_votes.append(0.85)
            weights.append(1.5)
            pitch_naturalness = 25.0
            indicators.append(
                f"Abnormally low pitch jitter ({jitter * 100:.2f}%) detected: synthetic voices often lack organic human micro-pitch modulation."
            )
        elif jitter > 0.065:
            # Phase discontinuity / neural vocoder glitching
            synthetic_votes.append(0.75)
            weights.append(1.2)
            pitch_naturalness = 35.0
            indicators.append(
                f"Elevated pitch jitter ({jitter * 100:.2f}%) detected: suggests phase inconsistency across synthesized frames."
            )
        else:
            # Natural human range
            synthetic_votes.append(0.15)
            weights.append(1.4)
            pitch_naturalness = 90.0
            indicators.append("Pitch stability and micro-jitter fall within typical human vocal cords range.")
    else:
        # Fallback if no voiced pitch segments found (e.g., whisper, noise, or pure unvoiced)
        pitch_naturalness = 60.0

    # 2. Spectral Consistency & High-Frequency Artifacts
    # Neural vocoders (HiFi-GAN, MelGAN) often cause high-frequency spectral aliasing.
    spectral_consistency = 80.0
    vocoder_artifact_score = 15.0

    centroid = features.spectral_centroid_hz
    rolloff = features.spectral_rolloff_hz
    flatness = features.spectral_flatness

    # Ratio of rolloff to centroid: typically 1.8x - 3.2x in natural human speech
    if centroid > 0:
        ratio = rolloff / centroid
        if ratio < 1.4 or ratio > 4.2:
            synthetic_votes.append(0.70)
            weights.append(1.1)
            spectral_consistency -= 35.0
            vocoder_artifact_score += 40.0
            indicators.append(
                f"Atypical spectral rolloff-to-centroid ratio ({ratio:.2f}) observed: suggests synthetic spectral shaping."
            )
        else:
            synthetic_votes.append(0.20)
            weights.append(1.0)
            spectral_consistency += 10.0

    # High spectral flatness in voice recording indicates artificial noise floor or phase cancellation
    if flatness > 0.15:
        synthetic_votes.append(0.80)
        weights.append(1.3)
        vocoder_artifact_score += 35.0
        indicators.append(
            f"High spectral flatness ({flatness:.4f}) detected: indicative of neural vocoder phase artifacts."
        )
    elif flatness < 0.05:
        synthetic_votes.append(0.18)
        weights.append(1.0)

    # 3. Vocal Tract Filter Dynamics & Energy Flow
    # Human voices possess dynamic transitions; synthetic voices often show frame-boundary rigidity.
    vocal_tract_dynamics = 82.0
    if features.zero_crossing_rate > 0.35:
        synthetic_votes.append(0.68)
        weights.append(0.8)
        vocal_tract_dynamics -= 25.0
        indicators.append("High zero-crossing rate detected, indicating high-frequency noise or vocoder residue.")
    elif features.zero_crossing_rate < 0.02:
        vocal_tract_dynamics -= 15.0

    # Aggregate weighted probability
    if weights:
        fake_probability = float(np.average(synthetic_votes, weights=weights))
    else:
        fake_probability = 0.50

    # Bound probability
    fake_probability = float(np.clip(fake_probability, 0.02, 0.98))

    # Calculate confidence score (distance from ambiguous 0.50 scaled to percentage)
    confidence = float(np.clip(50.0 + abs(fake_probability - 0.50) * 95.0, 50.0, 99.5))

    # Risk level classification
    if fake_probability >= 0.70:
        risk_level = RiskLevel.HIGH_RISK_DEEPFAKE
        is_synthetic = True
        summary = "High probability of AI-generated synthetic voice. Critical fraud prevention alert triggered."
    elif fake_probability >= 0.35:
        risk_level = RiskLevel.SUSPICIOUS
        is_synthetic = False
        summary = "Acoustic anomalies detected. Voice sample shows suspicious patterns; secondary verification recommended."
    else:
        risk_level = RiskLevel.LOW_RISK
        is_synthetic = False
        summary = "Acoustic profile aligns with authentic human voice characteristics. No deepfake patterns found."

    breakdown = AnalysisBreakdown(
        spectral_consistency=round(float(np.clip(spectral_consistency, 5.0, 98.0)), 1),
        pitch_naturalness=round(float(np.clip(pitch_naturalness, 5.0, 98.0)), 1),
        vocal_tract_dynamics=round(float(np.clip(vocal_tract_dynamics, 5.0, 98.0)), 1),
        vocoder_artifact_score=round(float(np.clip(vocoder_artifact_score, 2.0, 98.0)), 1),
    )

    return (
        round(fake_probability, 4),
        round(confidence, 1),
        risk_level,
        is_synthetic,
        summary,
        indicators,
        breakdown,
    )
