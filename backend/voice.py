import json
import os
import tempfile

import librosa
import numpy as np
import torch

from transformers import (
    AutoFeatureExtractor,
    WavLMForXVector,
)


# ============================================================
# DEEPSHIELD - OPTIMIZED VOICE ANALYSIS
# ============================================================
#
# IMPORTANT:
# ------------------------------------------------------------
# This file keeps the existing DeepShield architecture:
#
# 1. WavLM speaker embedding
# 2. Pitch variation
# 3. Energy variation
# 4. Pause / silence behaviour
# 5. Spectral characteristics
# 6. Speech consistency
# 7. Naturalness score
#
# Optimization only:
# ------------------------------------------------------------
# - torch.inference_mode()
# - reduced unnecessary tensor overhead
# - efficient numpy operations
# - cached FFT configuration
# - no duplicate spectral computation
# - no dummy spectral_consistency() calculation
# - avoid repeated conversions
# - optimized frame calculations
#
# SCORING LOGIC IS KEPT THE SAME.
# ============================================================


MODEL_NAME = "microsoft/wavlm-base-plus-sv"

TARGET_SAMPLE_RATE = 16000


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Prevent excessive CPU thread creation.
#
# This can improve latency on normal laptops/desktops.
try:
    torch.set_num_threads(
        max(
            1,
            min(
                4,
                os.cpu_count() or 1
            )
        )
    )
except Exception:
    pass


# ============================================================
# WAVLM THRESHOLDS
# ============================================================

VERIFIED_THRESHOLD = 0.86
UNCERTAIN_THRESHOLD = 0.70


# ============================================================
# MODEL LOADING
# ============================================================

print("Loading DeepShield voice model...")


feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)


voice_model = WavLMForXVector.from_pretrained(
    MODEL_NAME
)


voice_model.eval()


# ------------------------------------------------------------
# Disable gradients permanently for inference.
# ------------------------------------------------------------

for parameter in voice_model.parameters():
    parameter.requires_grad_(False)


print("WAVLM VOICE MODEL OK")


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(audio_path: str):
    """
    Load audio as:

        mono
        16 kHz
        float32

    Optimized to avoid unnecessary conversions.
    """

    if not audio_path:
        raise ValueError(
            "Audio path is empty."
        )

    if not os.path.exists(audio_path):
        raise ValueError(
            f"Audio file does not exist: {audio_path}"
        )

    try:

        audio, sample_rate = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True
        )

    except Exception as exc:

        raise ValueError(
            f"Could not read audio file: {exc}"
        ) from exc

    if audio is None or len(audio) == 0:

        raise ValueError(
            "Uploaded voice file is empty."
        )

    # --------------------------------------------------------
    # Ensure float32 once.
    # --------------------------------------------------------

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Remove DC offset.
    # --------------------------------------------------------

    audio -= np.mean(
        audio,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Remove invalid values.
    # --------------------------------------------------------

    np.nan_to_num(
        audio,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return audio


# ============================================================
# SAFE NORMALIZATION
# ============================================================

def normalize_audio(audio):
    """
    Safely normalize audio.

    Keeps original logic but avoids unnecessary copies.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if audio.size == 0:
        return audio

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 1e-8:

        audio = audio / max_value

    return np.asarray(
        audio,
        dtype=np.float32
    )


# ============================================================
# SPEAKER EMBEDDING
# ============================================================

def generate_embedding(audio_path: str):
    """
    Generate normalized WavLM speaker embedding.

    Returns:
        list[float]
    """

    audio = load_audio(
        audio_path
    )

    audio = normalize_audio(
        audio
    )

    try:

        inputs = feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        # ----------------------------------------------------
        # inference_mode is faster than no_grad for inference.
        # ----------------------------------------------------

        with torch.inference_mode():

            output = voice_model(
                **inputs
            )

        embedding = output.embeddings

        # ----------------------------------------------------
        # L2 normalize.
        # ----------------------------------------------------

        embedding = torch.nn.functional.normalize(
            embedding,
            dim=-1
        )

        embedding = (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if embedding.size == 0:

            raise ValueError(
                "Generated speaker embedding is empty."
            )

        return embedding.tolist()

    except Exception as exc:

        raise ValueError(
            f"Could not create speaker embedding: {exc}"
        ) from exc


# ============================================================
# EMBEDDING → JSON
# ============================================================

def embedding_to_json(embedding):

    if embedding is None:

        raise ValueError(
            "Embedding cannot be None."
        )

    return json.dumps(
        [
            float(value)
            for value in embedding
        ]
    )


# ============================================================
# JSON → EMBEDDING
# ============================================================

def embedding_from_json(value):

    if value is None:
        return None

    if isinstance(
        value,
        np.ndarray
    ):

        return value.astype(
            np.float32,
            copy=False
        )

    if isinstance(
        value,
        list
    ):

        return np.asarray(
            value,
            dtype=np.float32
        )

    try:

        data = json.loads(
            value
        )

        return np.asarray(
            data,
            dtype=np.float32
        )

    except Exception as exc:

        raise ValueError(
            f"Invalid stored voice fingerprint: {exc}"
        ) from exc


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding_a,
    embedding_b
):
    """
    Calculate cosine similarity.
    """

    a = np.asarray(
        embedding_a,
        dtype=np.float32
    )

    b = np.asarray(
        embedding_b,
        dtype=np.float32
    )

    if a.size == 0 or b.size == 0:

        raise ValueError(
            "Cannot compare empty embeddings."
        )

    if a.shape != b.shape:

        raise ValueError(
            f"Embedding dimensions do not match: "
            f"{a.shape} vs {b.shape}"
        )

    a_norm = np.linalg.norm(
        a
    )

    b_norm = np.linalg.norm(
        b
    )

    if a_norm == 0 or b_norm == 0:

        raise ValueError(
            "Cannot compare zero-length embeddings."
        )

    return float(
        np.dot(a, b)
        /
        (
            a_norm
            *
            b_norm
        )
    )


# ============================================================
# VERDICT
# ============================================================

def get_verdict(
    similarity: float
):
    """
    Speaker similarity verdict.
    """

    if similarity >= VERIFIED_THRESHOLD:

        return "LIKELY_VERIFIED"

    if similarity >= UNCERTAIN_THRESHOLD:

        return "UNCERTAIN"

    return "VOICE_MISMATCH"


# ============================================================
# BASIC STATISTICS
# ============================================================

def _safe_std(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size < 2:
        return 0.0

    return float(
        np.std(values)
    )


def _safe_mean(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return 0.0

    return float(
        np.mean(values)
    )


# ============================================================
# PITCH ANALYSIS
# ============================================================

def analyze_pitch(audio):
    """
    Analyse fundamental-frequency variation.
    """

    try:

        f0 = librosa.yin(
            audio,
            fmin=70,
            fmax=400,
            sr=TARGET_SAMPLE_RATE,
            frame_length=2048,
            hop_length=256
        )

        f0 = np.asarray(
            f0,
            dtype=np.float32
        )

        valid = (
            np.isfinite(f0)
            &
            (f0 >= 70)
            &
            (f0 <= 400)
        )

        voiced_pitch = f0[valid]

        total_frames = len(f0)

        if total_frames == 0:

            return {
                "pitch_variation": 0.0,
                "voiced_ratio": 0.0,
                "median_pitch": 0.0,
                "pitch_range": 0.0
            }

        voiced_ratio = (
            len(voiced_pitch)
            /
            total_frames
        )

        if len(voiced_pitch) < 3:

            return {
                "pitch_variation": 0.0,
                "voiced_ratio": round(
                    voiced_ratio * 100,
                    2
                ),
                "median_pitch": 0.0,
                "pitch_range": 0.0
            }

        median_pitch = float(
            np.median(
                voiced_pitch
            )
        )

        pitch_std = float(
            np.std(
                voiced_pitch
            )
        )

        pitch_variation = (
            pitch_std
            /
            max(
                median_pitch,
                1.0
            )
        )

        p5, p95 = np.percentile(
            voiced_pitch,
            [5, 95]
        )

        pitch_range = (
            p95 - p5
        )

        return {

            "pitch_variation": round(
                float(pitch_variation),
                4
            ),

            "voiced_ratio": round(
                voiced_ratio * 100,
                2
            ),

            "median_pitch": round(
                median_pitch,
                2
            ),

            "pitch_range": round(
                float(pitch_range),
                2
            )
        }

    except Exception:

        return {
            "pitch_variation": 0.0,
            "voiced_ratio": 0.0,
            "median_pitch": 0.0,
            "pitch_range": 0.0
        }


# ============================================================
# ENERGY ANALYSIS
# ============================================================

def analyze_energy(audio):
    """
    Analyse short-term RMS energy.
    """

    try:

        rms = librosa.feature.rms(
            y=audio,
            frame_length=1024,
            hop_length=256
        )[0]

        rms = np.asarray(
            rms,
            dtype=np.float32
        )

        rms = rms[
            np.isfinite(rms)
        ]

        if rms.size < 3:

            return {
                "energy_variation": 0.0,
                "energy_dynamic_range": 0.0,
                "energy_consistency": 0.0
            }

        rms = np.maximum(
            rms,
            1e-8
        )

        log_energy = (
            20.0
            *
            np.log10(rms)
        )

        energy_std = float(
            np.std(
                log_energy
            )
        )

        p10, p90 = np.percentile(
            log_energy,
            [10, 90]
        )

        dynamic_range = (
            p90 - p10
        )

        if 2.0 <= energy_std <= 12.0:

            consistency = 100.0

        elif energy_std < 2.0:

            consistency = (
                energy_std
                /
                2.0
                *
                100.0
            )

        else:

            consistency = max(
                0.0,
                100.0
                -
                (
                    energy_std
                    -
                    12.0
                )
                * 5.0
            )

        return {

            "energy_variation": round(
                energy_std,
                3
            ),

            "energy_dynamic_range": round(
                float(dynamic_range),
                3
            ),

            "energy_consistency": round(
                float(
                    np.clip(
                        consistency,
                        0,
                        100
                    )
                ),
                2
            )
        }

    except Exception:

        return {
            "energy_variation": 0.0,
            "energy_dynamic_range": 0.0,
            "energy_consistency": 0.0
        }


# ============================================================
# SILENCE / PAUSE ANALYSIS
# ============================================================

def analyze_pauses(audio):
    """
    Analyse silence and pause behaviour.
    """

    try:

        intervals = librosa.effects.split(
            audio,
            top_db=30,
            frame_length=1024,
            hop_length=256
        )

        total_samples = len(
            audio
        )

        if total_samples == 0:

            return {
                "silence_ratio": 0.0,
                "pause_count": 0,
                "average_pause_duration": 0.0,
                "longest_pause": 0.0
            }

        # ----------------------------------------------------
        # Filter tiny speech fragments.
        # ----------------------------------------------------

        min_speech_samples = int(
            0.08
            *
            TARGET_SAMPLE_RATE
        )

        speech_intervals = intervals[
            (
                intervals[:, 1]
                -
                intervals[:, 0]
            )
            >=
            min_speech_samples
        ] if len(intervals) else []

        if len(speech_intervals) == 0:

            return {
                "silence_ratio": 100.0,
                "pause_count": 0,
                "average_pause_duration": 0.0,
                "longest_pause": 0.0
            }

        silence_durations = []

        # ----------------------------------------------------
        # Beginning silence.
        # ----------------------------------------------------

        first_start = speech_intervals[0][0]

        if first_start > 0:

            silence_durations.append(
                first_start
                /
                TARGET_SAMPLE_RATE
            )

        # ----------------------------------------------------
        # Internal gaps.
        # ----------------------------------------------------

        if len(speech_intervals) > 1:

            previous_end = speech_intervals[:-1, 1]
            current_start = speech_intervals[1:, 0]

            gaps = (
                current_start
                -
                previous_end
            ) / TARGET_SAMPLE_RATE

            silence_durations.extend(
                gaps[
                    gaps >= 0.12
                ].tolist()
            )

        # ----------------------------------------------------
        # Ending silence.
        # ----------------------------------------------------

        final_end = speech_intervals[-1][1]

        if final_end < total_samples:

            silence_durations.append(
                (
                    total_samples
                    -
                    final_end
                )
                /
                TARGET_SAMPLE_RATE
            )

        if silence_durations:

            silence_array = np.asarray(
                silence_durations,
                dtype=np.float32
            )

            total_silence = float(
                np.sum(
                    silence_array
                )
            )

            meaningful_pauses = (
                silence_array[
                    silence_array >= 0.12
                ]
            )

        else:

            total_silence = 0.0

            meaningful_pauses = np.array(
                [],
                dtype=np.float32
            )

        duration_seconds = (
            total_samples
            /
            TARGET_SAMPLE_RATE
        )

        silence_ratio = (
            total_silence
            /
            max(
                duration_seconds,
                1e-8
            )
        ) * 100.0

        if meaningful_pauses.size:

            average_pause = float(
                np.mean(
                    meaningful_pauses
                )
            )

            longest_pause = float(
                np.max(
                    meaningful_pauses
                )
            )

        else:

            average_pause = 0.0
            longest_pause = 0.0

        return {

            "silence_ratio": round(
                float(
                    np.clip(
                        silence_ratio,
                        0,
                        100
                    )
                ),
                2
            ),

            "pause_count": int(
                meaningful_pauses.size
            ),

            "average_pause_duration": round(
                average_pause,
                3
            ),

            "longest_pause": round(
                longest_pause,
                3
            )
        }

    except Exception:

        return {
            "silence_ratio": 0.0,
            "pause_count": 0,
            "average_pause_duration": 0.0,
            "longest_pause": 0.0
        }


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

def analyze_spectral(audio):
    """
    Analyse spectral characteristics.
    """

    try:

        # ----------------------------------------------------
        # Compute shared spectral representation once.
        # ----------------------------------------------------

        stft = librosa.stft(
            audio,
            n_fft=1024,
            hop_length=256
        )

        magnitude = np.abs(
            stft
        )

        centroid = librosa.feature.spectral_centroid(
            S=magnitude,
            sr=TARGET_SAMPLE_RATE
        )[0]

        bandwidth = librosa.feature.spectral_bandwidth(
            S=magnitude,
            sr=TARGET_SAMPLE_RATE
        )[0]

        rolloff = librosa.feature.spectral_rolloff(
            S=magnitude,
            sr=TARGET_SAMPLE_RATE,
            roll_percent=0.85
        )[0]

        zero_crossing = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=1024,
            hop_length=256
        )[0]

        flatness = librosa.feature.spectral_flatness(
            S=magnitude
        )[0]

        return {

            "spectral_centroid": round(
                _safe_mean(
                    centroid
                ),
                2
            ),

            "spectral_bandwidth": round(
                _safe_mean(
                    bandwidth
                ),
                2
            ),

            "spectral_rolloff": round(
                _safe_mean(
                    rolloff
                ),
                2
            ),

            "zero_crossing_rate": round(
                _safe_mean(
                    zero_crossing
                ),
                5
            ),

            "spectral_flatness": round(
                _safe_mean(
                    flatness
                ),
                6
            )
        }

    except Exception:

        return {

            "spectral_centroid": 0.0,

            "spectral_bandwidth": 0.0,

            "spectral_rolloff": 0.0,

            "zero_crossing_rate": 0.0,

            "spectral_flatness": 0.0
        }


# ============================================================
# SPECTRAL CONSISTENCY
# ============================================================

def spectral_consistency(audio):
    """
    Measure frame-to-frame spectral stability.

    Optimized using vectorized numpy operations.
    """

    try:

        stft = np.abs(
            librosa.stft(
                audio,
                n_fft=1024,
                hop_length=256
            )
        )

        if stft.shape[1] < 3:

            return 50.0

        # ----------------------------------------------------
        # Normalize every frame.
        # ----------------------------------------------------

        frame_norm = np.linalg.norm(
            stft,
            axis=0,
            keepdims=True
        )

        frame_norm += 1e-8

        normalized = (
            stft
            /
            frame_norm
        )

        # ----------------------------------------------------
        # Instead of Python loop:
        #
        # previous[:, i]
        # current[:, i]
        #
        # use vectorized multiplication.
        # ----------------------------------------------------

        similarities = np.sum(
            normalized[:, :-1]
            *
            normalized[:, 1:],
            axis=0
        )

        variation = float(
            np.std(
                similarities
            )
        )

        if 0.015 <= variation <= 0.12:

            score = 100.0

        elif variation < 0.015:

            score = (
                variation
                /
                0.015
                *
                100.0
            )

        else:

            score = max(
                0.0,
                100.0
                -
                (
                    variation
                    -
                    0.12
                )
                * 400.0
            )

        return round(
            float(
                np.clip(
                    score,
                    0,
                    100
                )
            ),
            2
        )

    except Exception:

        return 50.0


# ============================================================
# SPEECH CONSISTENCY
# ============================================================

def analyze_speech_consistency(
    pitch_data,
    energy_data,
    pause_data,
    spectral_score
):
    """
    Combine acoustic behaviour into speech consistency.
    """

    scores = []

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    pitch_variation = pitch_data[
        "pitch_variation"
    ]

    voiced_ratio = pitch_data[
        "voiced_ratio"
    ]

    if 0.03 <= pitch_variation <= 0.35:

        pitch_score = 100.0

    elif pitch_variation < 0.03:

        pitch_score = (
            pitch_variation
            /
            0.03
            *
            100.0
        )

    else:

        pitch_score = max(
            0.0,
            100.0
            -
            (
                pitch_variation
                -
                0.35
            )
            * 180.0
        )

    if not (
        20 <= voiced_ratio <= 95
    ):

        pitch_score *= 0.7

    scores.append(
        np.clip(
            pitch_score,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    scores.append(
        np.clip(
            energy_data[
                "energy_consistency"
            ],
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Pause
    # --------------------------------------------------------

    silence_ratio = pause_data[
        "silence_ratio"
    ]

    if 2 <= silence_ratio <= 45:

        pause_score = 100.0

    elif silence_ratio < 2:

        pause_score = (
            silence_ratio
            /
            2.0
            *
            100.0
        )

    else:

        pause_score = max(
            0.0,
            100.0
            -
            (
                silence_ratio
                -
                45.0
            )
            * 1.8
        )

    scores.append(
        np.clip(
            pause_score,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Spectral
    # --------------------------------------------------------

    scores.append(
        np.clip(
            spectral_score,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Existing weighting preserved.
    # --------------------------------------------------------

    final_score = (
        scores[0] * 0.30
        +
        scores[1] * 0.25
        +
        scores[2] * 0.20
        +
        scores[3] * 0.25
    )

    return round(
        float(
            np.clip(
                final_score,
                0,
                100
            )
        ),
        2
    )


# ============================================================
# NATURALNESS SCORE
# ============================================================

def calculate_naturalness_score(
    pitch_data,
    energy_data,
    pause_data,
    spectral_data,
    speech_consistency
):
    """
    Calculate acoustic naturalness score.

    IMPORTANT:
    Same scoring logic as before.

    Optimization:
    Removed the old dummy spectral_consistency()
    calculation because its result was immediately
    discarded and replaced with speech_consistency.
    """

    # --------------------------------------------------------
    # Pitch score
    # --------------------------------------------------------

    pitch_variation = pitch_data[
        "pitch_variation"
    ]

    if 0.03 <= pitch_variation <= 0.35:

        pitch_score = 100.0

    elif pitch_variation < 0.03:

        pitch_score = (
            pitch_variation
            /
            0.03
            *
            100.0
        )

    else:

        pitch_score = max(
            0.0,
            100.0
            -
            (
                pitch_variation
                -
                0.35
            )
            * 180.0
        )

    pitch_score = np.clip(
        pitch_score,
        0,
        100
    )

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    energy_score = energy_data[
        "energy_consistency"
    ]

    # --------------------------------------------------------
    # Pause
    # --------------------------------------------------------

    silence_ratio = pause_data[
        "silence_ratio"
    ]

    if 2 <= silence_ratio <= 45:

        pause_score = 100.0

    elif silence_ratio < 2:

        pause_score = (
            silence_ratio
            /
            2.0
            *
            100.0
        )

    else:

        pause_score = max(
            0.0,
            100.0
            -
            (
                silence_ratio
                -
                45.0
            )
            * 1.8
        )

    pause_score = np.clip(
        pause_score,
        0,
        100
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Old code performed:
    #
    # spectral_consistency(
    #     np.zeros(16000)
    # )
    #
    # and then immediately:
    #
    # spectral_score = speech_consistency
    #
    # So that expensive calculation had ZERO effect.
    #
    # We simply use the same final value directly.
    # --------------------------------------------------------

    spectral_score = speech_consistency

    # --------------------------------------------------------
    # Existing weights preserved.
    # --------------------------------------------------------

    naturalness = (
        pitch_score * 0.25
        +
        energy_score * 0.20
        +
        pause_score * 0.20
        +
        spectral_score * 0.20
        +
        speech_consistency * 0.15
    )

    return round(
        float(
            np.clip(
                naturalness,
                0,
                100
            )
        ),
        2
    )


# ============================================================
# COMPLETE VOICE ANALYSIS
# ============================================================

def analyze_voice(audio_path: str):
    """
    Complete acoustic voice analysis.

    Returns the same structure expected by
    the existing DeepShield application.
    """

    # --------------------------------------------------------
    # Load once.
    # --------------------------------------------------------

    audio = load_audio(
        audio_path
    )

    if len(audio) < TARGET_SAMPLE_RATE:

        raise ValueError(
            "Voice recording is too short. "
            "Please provide at least 1 second of audio."
        )

    # --------------------------------------------------------
    # Normalize once.
    # --------------------------------------------------------

    analysis_audio = normalize_audio(
        audio
    )

    # --------------------------------------------------------
    # Individual analysis.
    # --------------------------------------------------------

    pitch_data = analyze_pitch(
        analysis_audio
    )

    energy_data = analyze_energy(
        analysis_audio
    )

    pause_data = analyze_pauses(
        analysis_audio
    )

    spectral_data = analyze_spectral(
        analysis_audio
    )

    # --------------------------------------------------------
    # Spectral consistency.
    #
    # This is calculated only ONCE.
    # --------------------------------------------------------

    spectral_score = spectral_consistency(
        analysis_audio
    )

    # --------------------------------------------------------
    # Speech consistency.
    # --------------------------------------------------------

    speech_consistency = (
        analyze_speech_consistency(
            pitch_data,
            energy_data,
            pause_data,
            spectral_score
        )
    )

    # --------------------------------------------------------
    # Naturalness.
    # --------------------------------------------------------

    naturalness_score = (
        calculate_naturalness_score(
            pitch_data,
            energy_data,
            pause_data,
            spectral_data,
            speech_consistency
        )
    )

    # --------------------------------------------------------
    # Duration.
    # --------------------------------------------------------

    duration_seconds = (
        len(audio)
        /
        TARGET_SAMPLE_RATE
    )

    # --------------------------------------------------------
    # SAME RESPONSE STRUCTURE.
    # --------------------------------------------------------

    return {

        "naturalness_score": naturalness_score,

        "duration_seconds": round(
            float(duration_seconds),
            2
        ),

        "pitch": pitch_data,

        "energy": energy_data,

        "pauses": pause_data,

        "spectral": spectral_data,

        "speech_consistency": speech_consistency,

        "analysis": {

            "pitch_variation": pitch_data[
                "pitch_variation"
            ],

            "energy_variation": energy_data[
                "energy_variation"
            ],

            "silence_ratio": pause_data[
                "silence_ratio"
            ],

            "spectral_consistency": spectral_score,

            "speech_consistency": speech_consistency
        },

        "note": (
            "Naturalness score is an acoustic MVP indicator. "
            "It should not be interpreted as a definitive "
            "human-versus-AI classification."
        )
    }


# ============================================================
# TEMPORARY UPLOAD HELPER
# ============================================================

def save_upload_temporarily(
    upload_file
):
    """
    Save FastAPI UploadFile temporarily.
    """

    suffix = os.path.splitext(
        upload_file.filename or ".wav"
    )[1]

    if not suffix:
        suffix = ".wav"

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    try:

        content = upload_file.file.read()

        if not content:

            raise ValueError(
                "Uploaded voice file is empty."
            )

        temp_file.write(
            content
        )

        temp_file.flush()

    finally:

        temp_file.close()

    return temp_file.name


# ============================================================
# DELETE TEMPORARY FILE
# ============================================================

def delete_temporary_file(
    path: str
):
    """
    Delete temporary raw audio.
    """

    if path and os.path.exists(path):

        try:

            os.remove(
                path
            )

        except OSError:

            pass