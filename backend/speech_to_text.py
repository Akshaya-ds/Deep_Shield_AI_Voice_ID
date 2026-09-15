import os

from faster_whisper import WhisperModel


# ============================================================
# DeepShield Speech-to-Text
# FAST CPU CONFIGURATION
# ============================================================

MODEL_SIZE = "tiny.en"

CPU_THREADS = max(
    4,
    min(6, os.cpu_count() or 4)
)

print("Loading optimized Faster-Whisper model...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=CPU_THREADS,
    num_workers=1
)

print(
    f"FASTER WHISPER READY | "
    f"model={MODEL_SIZE} | "
    f"threads={CPU_THREADS}"
)


# ============================================================
# TRANSCRIBE AUDIO
# ============================================================

def transcribe_audio(audio_path: str) -> str:

    segments, info = model.transcribe(

        audio_path,

        # Fast decoding
        beam_size=1,

        # English banking conversations
        language="en",

        # Remove unnecessary silence
        vad_filter=True,

        # Faster independent segments
        condition_on_previous_text=False,

        # No alternative candidates
        best_of=1,

        # Avoid silence hallucination
        no_speech_threshold=0.6,

        # Faster VAD
        vad_parameters={
            "min_silence_duration_ms": 500
        }
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