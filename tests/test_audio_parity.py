import numpy as np

from assistant.dsp import LogMelExtractor
from assistant.wakeword.onnx_detector import OnnxWakeWordDetector


def generate_test_signals(sr: int, n: int):
    """Deterministic synthetic signals to exercise the pipeline."""
    t = np.arange(n) / sr

    return {
        "silence": np.zeros(n, dtype=np.int16),
        "impulse": np.pad(np.array([32767], dtype=np.int16), (0, n - 1)),
        "sine_1khz": (0.5 * np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16),
        "quiet_noise": (np.random.default_rng(0).normal(0, 500, n)).astype(np.int16),
        "loud_noise": (np.random.default_rng(1).normal(0, 15000, n)).astype(np.int16),
    }


def runtime_preprocess(pcm: np.ndarray) -> np.ndarray:
    """Mirror OnnxWakeWordDetector preprocessing."""
    pcm_f = pcm.astype(np.float32)
    pcm_f *= 2.5
    pcm_f = np.tanh(pcm_f / 20000.0) * 20000.0
    return pcm_f.astype(np.int16)


def training_preprocess(pcm: np.ndarray) -> np.ndarray:
    """Mirror train_wakeword.py preprocessing."""
    pcm_f = pcm.astype(np.float32)
    pcm_f *= 2.5
    pcm_f = np.tanh(pcm_f / 20000.0) * 20000.0
    return pcm_f.astype(np.int16)


def test_runtime_training_logmel_parity():
    sr = 16000
    clip_seconds = 1.0
    n = int(sr * clip_seconds)

    extractor = LogMelExtractor(
        sr=sr,
        n_fft=1024,
        win_ms=25,
        hop_ms=10,
        n_mels=40,
        clip_seconds=clip_seconds,
    )

    signals = generate_test_signals(sr, n)

    for name, pcm in signals.items():
        rt_pcm = runtime_preprocess(pcm.copy())
        tr_pcm = training_preprocess(pcm.copy())

        rt_logmel = extractor.extract(rt_pcm)
        tr_logmel = extractor.extract(tr_pcm)

        # ---- shape invariants ----
        assert rt_logmel.shape == tr_logmel.shape, f"{name}: shape mismatch"

        # ---- numerical invariants ----
        diff = np.mean(np.abs(rt_logmel - tr_logmel))
        assert diff < 1e-4, f"{name}: log-mel mismatch {diff}"
