import io
import wave
import numpy as np
from scipy.io import wavfile
from fastapi import HTTPException, status

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
MIN_DURATION_SECONDS = 0.25
SUPPORTED_EXTENSIONS = {".wav", ".wave", ".mp3", ".ogg", ".flac"}


def validate_and_load_audio(file_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    """
    Validates uploaded audio file and decodes it into a normalized single-channel
    float32 numpy array with its sampling rate.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty."
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)}MB."
        )

    # Validate file extension
    lower_filename = filename.lower()
    has_valid_extension = any(lower_filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS)
    if not has_valid_extension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    audio_array = None
    sample_rate = None

    # Primary decoder: scipy.io.wavfile
    try:
        sample_rate, raw_data = wavfile.read(io.BytesIO(file_bytes))
        audio_array = raw_data
    except Exception:
        # Fallback decoder: standard library wave module
        try:
            with wave.open(io.BytesIO(file_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

                if sampwidth == 1:
                    raw_data = np.frombuffer(frames, dtype=np.uint8)
                elif sampwidth == 2:
                    raw_data = np.frombuffer(frames, dtype=np.int16)
                elif sampwidth == 4:
                    raw_data = np.frombuffer(frames, dtype=np.int32)
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                if n_channels > 1:
                    raw_data = raw_data.reshape(-1, n_channels)
                audio_array = raw_data
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unable to decode audio payload. Ensure the audio is a valid uncompressed or standard WAV file: {str(exc)}"
            )

    # Convert multichannel to mono by averaging channels
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # Normalize audio values to float32 range [-1.0, 1.0]
    if np.issubdtype(audio_array.dtype, np.integer):
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        elif audio_array.dtype == np.uint8:
            audio_array = (audio_array.astype(np.float32) - 128.0) / 128.0
        else:
            max_val = np.iinfo(audio_array.dtype).max
            audio_array = audio_array.astype(np.float32) / float(max_val)
    else:
        audio_array = audio_array.astype(np.float32)
        audio_array = np.clip(audio_array, -1.0, 1.0)

    duration = len(audio_array) / float(sample_rate)
    if duration < MIN_DURATION_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration ({duration:.2f}s) is too short for acoustic forensic analysis. Minimum required is {MIN_DURATION_SECONDS}s."
        )

    return audio_array, int(sample_rate)
