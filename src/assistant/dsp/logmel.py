from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import stft


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    # Build mel filterbank matrix: (n_mels, n_fft//2 + 1)
    fmin = max(0.0, fmin)
    fmax = min(sr / 2.0, fmax)

    mels = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if right <= left:
            continue
        # rising slope
        for k in range(left, center):
            if 0 <= k < fb.shape[1] and center != left:
                fb[i - 1, k] = (k - left) / (center - left)
        # falling slope
        for k in range(center, right):
            if 0 <= k < fb.shape[1] and right != center:
                fb[i - 1, k] = (right - k) / (right - center)
    return fb


@dataclass
class LogMelExtractor:
    sr: int
    n_fft: int
    win_ms: int
    hop_ms: int
    n_mels: int
    fmin: float
    fmax: float
    log_eps: float
    clip_seconds: float

    def __post_init__(self):
        self.win_length = int(self.sr * self.win_ms / 1000.0)
        self.hop_length = int(self.sr * self.hop_ms / 1000.0)
        self.num_samples = int(self.sr * self.clip_seconds)
        self.num_frames = 1 + (self.num_samples - self.win_length) // self.hop_length

        self._fb = _mel_filterbank(self.sr, self.n_fft, self.n_mels, self.fmin, self.fmax)

    def extract(self, samples_i16: np.ndarray) -> np.ndarray:
        """
        Input: int16 mono samples length == num_samples
        Output: float32 array shaped (1, n_mels, num_frames) with log-mel energies
        """
        if samples_i16.dtype != np.int16:
            samples_i16 = samples_i16.astype(np.int16)

        x = samples_i16.astype(np.float32) / 32768.0

        # STFT magnitude^2
        _, _, Zxx = stft(
            x,
            fs=self.sr,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            padded=False,
            boundary=None,
        )
        # Zxx: (freq_bins, frames)
        power = (np.abs(Zxx) ** 2).astype(np.float32)

        # Mel projection
        mel = np.matmul(self._fb, power)  # (n_mels, frames)
        mel = np.log(mel + float(self.log_eps))

        # Ensure fixed frame count (pad/trim)
        frames = mel.shape[1]
        if frames < self.num_frames:
            pad = self.num_frames - frames
            mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
        elif frames > self.num_frames:
            mel = mel[:, : self.num_frames]

        return mel[np.newaxis, :, :].astype(np.float32)
