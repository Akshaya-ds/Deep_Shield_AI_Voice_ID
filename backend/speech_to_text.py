from faster_whisper import WhisperModel


# ============================================================
# DeepShield Speech-to-Text
# Optimized Faster-Whisper Configuration
# ============================================================

MODEL_SIZE = "tiny"

print("Loading optimized Faster-Whisper model...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1
)

print("FASTER WHISPER OPTIMIZED MODEL OK")


# ============================================================
# TRANSCRIBE AUDIO
# ============================================================

def transcribe_audio(audio_path: str) -> str:
    """
    Fast speech-to-text for DeepShield.

    Optimizations:
    - tiny model
    - int8 CPU inference
    - beam_size=1
    - VAD filtering
    - no previous-text conditioning

    Returns:
        transcript string
    """

    segments, info = model.transcribe(
        audio_path,

        # Faster decoding
        beam_size=1,

        # English banking calls
        language="en",

        # Remove long silence before transcription
        vad_filter=True,

        # Faster / more independent segments
        condition_on_previous_text=False,

        # Avoid unnecessary alternatives
        best_of=1,

        # Prevent hallucinated text during silence
        no_speech_threshold=0.6
    )

    transcript_parts = []

    for segment in segments:

        text = segment.text.strip()

        if text:
            transcript_parts.append(text)

    transcript = " ".join(
        transcript_parts
    ).strip()

    if not transcript:
        raise ValueError(
            "Could not extract speech from audio."
        )

    return transcript