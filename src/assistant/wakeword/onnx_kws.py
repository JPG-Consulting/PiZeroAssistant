from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import onnxruntime as ort


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@dataclass
class OnnxWakeword:
    model_path: str
    input_name: str = "logmel"
    output_name: str = "logits"
    threshold: float = 0.80
    consecutive_hits: int = 2
    cooldown_ms: int = 1500

    def __post_init__(self):
        self._sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self._hits = 0
        self._cooldown_until = 0.0

    def score(self, logmel: np.ndarray) -> float:
        """
        logmel: (1, n_mels, frames) float32
        Model expects (1,1,n_mels,frames)
        Returns p(wake)
        """
        if logmel.ndim != 3:
            raise ValueError(f"logmel must be (1,n_mels,frames), got {logmel.shape}")

        x = logmel[:, np.newaxis, :, :].astype(np.float32)  # (1,1,n_mels,frames)
        out = self._sess.run(None, {self.input_name: x})

        # Handle output indexing robustly
        logits = None
        for i, o in enumerate(self._sess.get_outputs()):
            if o.name == self.output_name:
                logits = out[i]
                break
        if logits is None:
            logits = out[0]

        probs = _softmax(np.asarray(logits, dtype=np.float32))
        # assume class 1 == wake (as per training in the Colab outline)
        return float(probs[0, 1])

    def update(self, p_wake: float) -> bool:
        """
        Apply smoothing/consecutive hits + cooldown.
        Returns True if this update emits a wake trigger.
        """
        now = time.time()
        if now < self._cooldown_until:
            self._hits = 0
            return False

        if p_wake >= self.threshold:
            self._hits += 1
        else:
            self._hits = 0

        if self._hits >= self.consecutive_hits:
            self._hits = 0
            self._cooldown_until = now + (self.cooldown_ms / 1000.0)
            return True

        return False
