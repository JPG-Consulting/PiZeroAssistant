# Google Colab Notebook — Wake-word Training (DS-CNN baseline → ONNX opset 11)

> **Objetivo**: entrenar un modelo *baseline* de wake-word (binario: `wake` vs `not_wake`) usando tus WAVs (16 kHz mono) y exportarlo a **ONNX compatible con Raspberry Pi** (**opset 11**, IR ≤ 11).
>
> **Entrada del modelo**: log-mel `(1, 40, 98)` (≈ 1.0 s con hop 10 ms).  
> **Salida del modelo**: probabilidad `p_wake` en `(1,)`.

---

## ✅ Antes de empezar (requisitos)

- Ya tienes tu dataset segmentado como:
  ```text
  dataset/
  ├── wake/
  └── not_wake/
  ```
- Has subido `dataset.zip` a Google Drive, por ejemplo:
  ```text
  MyDrive/wakeword-training/dataset.zip
  ```

> Si tu ruta es distinta, solo cambia `ZIP_PATH` en la celda correspondiente.

---

## 0) Configuración (parámetros congelados)

Estos parámetros deben **coincidir** con tu runtime del Pi:

- `sr = 16000`
- `clip_seconds = 1.0`
- `n_mels = 40`
- `win_ms = 25`
- `hop_ms = 10`
- `n_fft = 1024`
- `fmin = 20`
- `fmax = 7600`
- `log_eps = 1e-6`

---

# 🧩 CELDAS (copiar/pegar en Colab)

## Celda 1 — Instalar dependencias

**Qué hace:** instala librerías para audio, entrenamiento (PyTorch) y export a ONNX.

```bash
pip -q install numpy==2.* torch torchvision torchaudio torchcodec onnx onnxruntime onnxscript
```

---

## Celda 2 — Montar Google Drive

**Qué hace:** monta tu Google Drive en `/content/drive`.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Celda 3 — Descomprimir `dataset.zip` desde Drive

**Qué hace:** extrae `dataset.zip` a `/content/dataset`.

✅ **Acción requerida:** ajusta `ZIP_PATH` si tu archivo está en otra carpeta.

```python
import zipfile
from pathlib import Path

ZIP_PATH = Path("/content/drive/MyDrive/wakeword-training/dataset.zip")
OUT_DIR  = Path("/content/dataset")

assert ZIP_PATH.exists(), f"No existe: {ZIP_PATH}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(OUT_DIR)

print("Dataset extraído en:", OUT_DIR)
print("Contenido:", [p.name for p in OUT_DIR.iterdir()])
```

**Esperado:**
- `/content/dataset/wake`
- `/content/dataset/not_wake`

---

## Celda 4 — Sanity check del dataset

**Qué hace:** cuenta archivos y verifica que hay WAVs en ambas clases.

```python
from pathlib import Path

OUT_DIR = Path("/content/dataset")

# Detecta si hay un nivel extra "dataset/"
candidate_roots = [
    OUT_DIR,                 # /content/dataset
    OUT_DIR / "dataset",     # /content/dataset/dataset
]

ROOT = None
for r in candidate_roots:
    if (r / "wake").exists() and (r / "not_wake").exists():
        ROOT = r
        break

assert ROOT is not None, f"No encuentro wake/ y not_wake/ en {candidate_roots}"

wake_dir = ROOT / "wake"
neg_dir  = ROOT / "not_wake"

wake_wavs = sorted(wake_dir.glob("*.wav"))
neg_wavs  = sorted(neg_dir.glob("*.wav"))

print("Dataset root:", ROOT)
print("Wake:", len(wake_wavs))
print("Not_wake:", len(neg_wavs))

assert len(wake_wavs) > 0, "No hay WAVs en wake/"
assert len(neg_wavs) > 0, "No hay WAVs en not_wake/"
```

---

## Celda 5 — Validación ligera de audio (sr/mono/duración)

**Qué hace:** revisa un subconjunto aleatorio para detectar problemas típicos (sr ≠ 16k, stereo, duración < 1s).

```python
import random, wave
from pathlib import Path

def wav_info(p: Path):
    with wave.open(str(p), "rb") as w:
        return {
            "channels": w.getnchannels(),
            "sr": w.getframerate(),
            "sampwidth": w.getsampwidth(),
            "frames": w.getnframes(),
        }

def check_some(wavs, n=25):
    picks = random.sample(wavs, min(n, len(wavs)))
    bad = 0
    for p in picks:
        info = wav_info(p)
        dur = info["frames"] / float(info["sr"])
        issues = []
        if info["sr"] != 16000:
            issues.append(f"sr={info['sr']}")
        if info["channels"] != 1:
            issues.append(f"ch={info['channels']}")
        if info["sampwidth"] != 2:
            issues.append(f"sampwidth={info['sampwidth']}")
        if dur < 1.0:
            issues.append(f"dur={dur:.2f}s")
        if issues:
            bad += 1
            print("[WARN]", p.name, "->", ", ".join(issues))
    return bad

bad_w = check_some(wake_wavs, n=30)
bad_n = check_some(neg_wavs, n=30)
print("Bad wake samples:", bad_w, "Bad neg samples:", bad_n)
```

> Si aparecen muchos warnings, conviene arreglarlo **antes** de entrenar (resample/mono/export PCM16).

---

## Celda 6 — Dataset PyTorch + extracción log-mel (PARIDAD con runtime)

**Qué hace:** define una extracción de log-mel con `torchaudio` y crea un `Dataset` que devuelve:
- `X`: tensor `float32` con shape `(1, 40, 98)`
- `y`: label (0=not_wake, 1=wake)

✅ **Acción requerida:** si tu `clip_seconds` no es 1.0, ajusta `CLIP_SEC` (pero idealmente no lo cambies).

```python
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Parámetros (deben coincidir con runtime)
SR = 16000
N_MELS = 40
WIN_MS = 25
HOP_MS = 10
N_FFT = 1024
FMIN = 20.0
FMAX = 7600.0
LOG_EPS = 1e-6
CLIP_SEC = 1.0

WIN_LEN = int(SR * WIN_MS / 1000.0)
HOP_LEN = int(SR * HOP_MS / 1000.0)
NUM_SAMPLES = int(SR * CLIP_SEC)

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    win_length=WIN_LEN,
    hop_length=HOP_LEN,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    power=2.0,
)

def load_wav_mono_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))  # (ch, n)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    # asegurar longitud exacta
    n = wav.size(1)
    if n < NUM_SAMPLES:
        pad = NUM_SAMPLES - n
        wav = torch.nn.functional.pad(wav, (0, pad))
    elif n > NUM_SAMPLES:
        # centrado
        start = (n - NUM_SAMPLES) // 2
        wav = wav[:, start:start+NUM_SAMPLES]
    return wav  # (1, NUM_SAMPLES)

def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    m = mel(wav)  # (1, n_mels, frames)
    lm = torch.log(m + LOG_EPS)
    return lm

class WavClips(Dataset):
    def __init__(self, wake_paths, neg_paths):
        self.items = [(p, 1) for p in wake_paths] + [(p, 0) for p in neg_paths]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        wav = load_wav_mono_16k(p)
        x = wav_to_logmel(wav).to(torch.float32)  # (1, 40, frames)
        # Forzar frames esperados (≈98). Si difiere 1-2 frames por borde, recortamos/paddeamos.
        frames = x.size(-1)
        TARGET_FRAMES = 98
        if frames < TARGET_FRAMES:
            x = torch.nn.functional.pad(x, (0, TARGET_FRAMES - frames))
        elif frames > TARGET_FRAMES:
            x = x[..., :TARGET_FRAMES]
        return x, torch.tensor(y, dtype=torch.float32)

# Print shapes
tmp = WavClips(wake_wavs[:1], neg_wavs[:1])
x0, y0 = tmp[0]
print("x shape:", tuple(x0.shape), "y:", y0.item())
assert x0.shape == (1, 40, 98), f"Shape inesperado: {x0.shape}"
```

---

## Celda 7 — Split train/val (estratificado simple)

**Qué hace:** crea train/val (por defecto 90/10) y DataLoaders.

```python
import random
from torch.utils.data import DataLoader

random.seed(0)
wake_paths = wake_wavs.copy()
neg_paths  = neg_wavs.copy()
random.shuffle(wake_paths)
random.shuffle(neg_paths)

VAL_FRAC = 0.1
nw = len(wake_paths)
nn = len(neg_paths)

wake_val = wake_paths[:max(1, int(nw*VAL_FRAC))]
wake_tr  = wake_paths[max(1, int(nw*VAL_FRAC)):]

neg_val = neg_paths[:max(1, int(nn*VAL_FRAC))]
neg_tr  = neg_paths[max(1, int(nn*VAL_FRAC)):]

train_ds = WavClips(wake_tr, neg_tr)
val_ds   = WavClips(wake_val, neg_val)

BATCH = 64
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

print("Train:", len(train_ds), "Val:", len(val_ds))
```

---

## Celda 8 — Modelo DS-CNN pequeño (baseline)

**Qué hace:** define un modelo ligero para KWS que es fácil de exportar a ONNX.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.relu(x)

class TinyDSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (B, 1, 40, 98)
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.ds1 = DepthwiseSeparableConv(16, 32)
        self.ds2 = DepthwiseSeparableConv(32, 64)

        # Convertimos el mapa a (B, 64, 1, 1) con un Conv "global"
        self.global_conv = nn.Conv2d(64, 64, kernel_size=(40, 98), bias=False)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.global_conv(x)    # (B, 64, 1, 1)

        # ✅ Sin squeeze/view/reshape: usamos indexing fijo
        x = x[:, :, 0, 0]          # (B, 64)

        prob = torch.sigmoid(self.fc(x))  # (B, 1)
        return prob

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyDSCNN().to(device)
print("Device:", device)
```

---

## Celda 9 — Entrenamiento (baseline)

**Qué hace:** entrena con BCE y reporta loss/accuracy aproximada.

✅ **Acción requerida:** si ves sobreajuste (val empeora rápido), baja epochs o añade dropout más tarde.

```python
import torch
from tqdm import tqdm

def accuracy(pred, y):
    """
    pred: (B, 1) probabilities
    y   : (B, 1) float labels {0,1}
    """
    pred = pred.view(-1)
    y = y.view(-1)
    return ((pred >= 0.5) == (y >= 0.5)).float().mean().item()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = torch.nn.BCELoss()

EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    # -------- TRAIN --------
    model.train()
    tloss, tacc, n = 0.0, 0.0, 0

    for x, y in tqdm(train_dl, desc=f"train {epoch}/{EPOCHS}"):
        x = x.to(device)
        y = y.to(device).view(-1, 1)   # (B, 1)

        opt.zero_grad()
        p = model(x)                  # (B, 1)
        loss = bce(p, y)
        loss.backward()
        opt.step()

        bs = x.size(0)
        tloss += loss.item() * bs
        tacc  += accuracy(p.detach(), y.detach()) * bs
        n += bs

    tloss /= n
    tacc  /= n

    # -------- VALIDATION --------
    model.eval()
    vloss, vacc, n = 0.0, 0.0, 0

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            y = y.to(device).view(-1, 1)  # (B, 1)

            p = model(x)                 # (B, 1)
            loss = bce(p, y)

            bs = x.size(0)
            vloss += loss.item() * bs
            vacc  += accuracy(p, y) * bs
            n += bs

    vloss /= n
    vacc  /= n

    print(
        f"Epoch {epoch:02d} | "
        f"train loss {tloss:.4f} acc {tacc:.3f} | "
        f"val loss {vloss:.4f} acc {vacc:.3f}"
    )
```

---

## Celda 10 — Export a ONNX (export moderno -> post-proceso legacy)

**Qué hace:** exporta ONNX con entrada `logmel` y salida `prob`.

✅ **Acción requerida:** si quieres otro nombre, cambia `input_names` / `output_names`.

```python
import torch.onnx
from pathlib import Path

model.eval()
onnx_path = Path("/content/wakeword.onnx")

dummy = torch.zeros((1, 1, 40, 98), dtype=torch.float32, device=device)

torch.onnx.export(
    model,
    dummy,
    str(onnx_path),
    input_names=["logmel"],
    output_names=["prob"],
    opset_version=18,          # ← IMPORTANTE
    do_constant_folding=True,
    dynamic_axes=None,         # shapes fijos (mejor para Pi)
)

print("Exported:", onnx_path)
```

> Nota: el `IR version` lo fija la librería ONNX al guardar. Si luego el Pi se queja (IR>11), lo ajustaremos en el post-proceso (celda siguiente).

---

## Celda 11 — (Opcional) Forzar IR version compatible (≤ 11)

**Qué hace:** abre el ONNX y fuerza `ir_version=7` si fuera necesario.

```python
import onnx

m = onnx.load(str(onnx_path))

print("Before:")
print("  IR version:", m.ir_version)
print("  Opset:", m.opset_import[0].version)

# Forzar compatibilidad con ONNXRuntime ARM
m.ir_version = 7
for opset in m.opset_import:
    opset.version = 11

onnx.save(m, str(onnx_path))

print("After:")
print("  IR version:", m.ir_version)
print("  Opset:", m.opset_import[0].version)
```

---

## Celda 12 - Verificación rápida IR / opset (Opcional pero recomendado)

```python
import onnx

onnx_path = "/content/wakeword.onnx"

m = onnx.load(onnx_path)

print("ONNX metadata:")
print("  IR version   :", m.ir_version)
print("  Opset version:", m.opset_import[0].version)

assert m.ir_version <= 7, "IR version too new for Raspberry Pi"
assert m.opset_import[0].version <= 11, "Opset version too new for Raspberry Pi"

print("ONNX metadata compatible with Raspberry Pi")
```

---

## Celda 13 — Prueba rápida con ONNXRuntime en Colab

**Qué hace:** carga el modelo ONNX y comprueba que corre y devuelve prob.

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    str(onnx_path),
    providers=["CPUExecutionProvider"],
)

inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

print("I/O:", inp_name, "->", out_name)

x = np.zeros((1, 1, 40, 98), dtype=np.float32)
y = sess.run([out_name], {inp_name: x})[0]

print("Output shape:", y.shape)
print("Output value:", y)
```

---

## Celda 14 — Guardar el modelo en Drive o descargarlo

### Opción A: copiar a Drive (recomendado)

```python
from pathlib import Path
import shutil

SRC_DIR = Path("/content")
DST_DIR = Path("/content/drive/MyDrive/wakeword-training")

DST_DIR.mkdir(parents=True, exist_ok=True)

files_to_copy = [
    SRC_DIR / "wakeword.onnx",
    SRC_DIR / "wakeword.onnx.data",
]

for src in files_to_copy:
    assert src.exists(), f"Missing file: {src}"
    dst = DST_DIR / src.name
    shutil.copyfile(src, dst)
    print(f"Saved to: {dst}")

print("Both ONNX files copied to Google Drive")
```

### Opción B: descargar al PC

```python
from google.colab import files
from pathlib import Path

paths = [
    Path("/content/wakeword.onnx"),
    Path("/content/wakeword.onnx.data"),
]

for p in paths:
    assert p.exists(), f"Missing file: {p}"
    print("Downloading:", p.name)
    files.download(str(p))
```

---

# ✅ Siguiente paso (en el Raspberry Pi)

1. Copia `wakeword.onnx` a tu Pi en `models/` (FTP/SCP):
   ```text
   models/wakeword.onnx
   ```
2. En `config/config.yaml`:
   ```yaml
   wakeword:
     type: onnx
     model_path: models/wakeword.onnx
     input_name: logmel
     output_name: prob
     threshold: 0.8
     consecutive_hits: 2
     cooldown_ms: 1500
   ```
3. Ejecuta tu runtime:
   ```bash
   source .venv/bin/activate
   PYTHONPATH=src python -m assistant.main --config config/config.yaml
   ```

---

# Notas y mejoras (para la siguiente iteración)

- **Desbalance**: si tienes muchas más negativas, puedes:
  - reducir `neg_clips_per_file` en `make_clips.py`
  - o usar `WeightedRandomSampler` (lo añadimos si hace falta)
- **Augmentations** (más adelante):
  - mezcla con ruido
  - ganancia aleatoria
  - time shift
- **Cuantización** (más adelante):
  - INT8 para acelerar en Pi Zero 2 W
- **Métricas reales**:
  - medir distribución de `p_wake` en silencio/habla/wake
  - ajustar `threshold` y `consecutive_hits`

---

## Checklist de éxito (baseline)
- El modelo ONNX carga en el Pi sin errores de IR/opset.
- `p_wake` es bajo en silencio / habla normal.
- `p_wake` sube claramente al decir el wake-word.
- Puedes ajustar un umbral para separar ambos.

