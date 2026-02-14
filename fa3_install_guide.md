# FlashAttention-3 (FA3 / Hopper) offline install guide (H100 server)

This guide is written for a **multi-GPU server (H100 + L40S)** with **no public internet**, a Python **venv**, and an internal PyPI/Artifactory index. It captures the exact workflow we followed, including the common failure modes and how to fix them.

> FA3 (“hopper/”) is intended for **H100/H800 (SM90)** and requires **CUDA toolkit (nvcc) >= 12.3**. The upstream project recommends CUDA 12.8 for best performance, and installs FA3 from the repo’s `hopper/` directory via `python setup.py install`.  
> Sources: FlashAttention repo README. https://github.com/Dao-AILab/flash-attention

---

## 0) Quick checklist (what “success” looks like)

After installation, the following should be true:

- `import flash_attn_interface` works (this is the FA3 Python interface)  
- `import flashattn_hopper_cuda` works (this is the compiled CUDA extension `.so`)  
- Running on an H100 shows compute capability `(9, 0)`  
- A small `flash_attn_func()` call runs on GPU without errors  
- (Optional) `pytest -q -s test_flash_attn.py` passes from `flash-attention/hopper`

Upstream explicitly documents `import flash_attn_interface; flash_attn_interface.flash_attn_func()` and a `pytest` test command.  
Source: https://github.com/Dao-AILab/flash-attention

---

## 1) Identify an H100 and pin it reliably

Because your node also has L40S GPUs, **do not rely on GPU indices**. Prefer **GPU UUIDs**.

### 1.1 List GPUs and choose an H100 UUID
```bash
nvidia-smi -L
# Example output includes lines like:
# GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-83e5dd57-b016-f03a-03a5-50a385b1dcb6)
```

### 1.2 Pin the process to that H100 (recommended)
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-83e5dd57-b016-f03a-03a5-50a385b1dcb6   # <-- your H100 UUID
```

NVIDIA documents that `CUDA_VISIBLE_DEVICES` can be set to GPU UUIDs (and even abbreviated unique prefixes), and recommends UUIDs to avoid ambiguity.  
Sources:  
- https://docs.nvidia.com/deploy/topics/topic_5_2_1.html  
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html

### 1.3 Sanity-check you are truly on H100
```bash
python - <<'PY'
import torch
print("GPU:", torch.cuda.get_device_name(0))
print("CC:", torch.cuda.get_device_capability(0))
PY
# Expect: GPU: NVIDIA H100 ...  CC: (9, 0)
```

---

## 2) Ensure CUDA toolkit (nvcc) is available (build-time requirement)

FA3 compilation needs the **CUDA toolkit compiler** (`nvcc`). Your PyTorch wheel provides CUDA runtime libraries, but not necessarily `nvcc` on PATH.

### 2.1 Verify nvcc exists
```bash
ls -l /usr/local/cuda/bin/nvcc
/usr/local/cuda/bin/nvcc -V
```

### 2.2 Put nvcc on PATH for the build
```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
nvcc -V
```

### 2.3 Version requirement
FA3 is only supported on **CUDA >= 12.3**; upstream’s `hopper/setup.py` checks the nvcc version and will error out otherwise.  
Source: https://github.com/Dao-AILab/flash-attention/issues/1477

---

## 3) Confirm your Python environment

Activate your venv and confirm PyTorch is visible:

```bash
source /path/to/unlearning-venv/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
PY
```

> You may see warnings about “detected CUDA version (12.x) has a minor version mismatch with PyTorch (12.1)”. This is typically a warning rather than a hard error when building extensions.  
> Source discussion: https://discuss.pytorch.org/t/pythorch-version-for-cuda-12-2/195217

---

## 4) Get FA3 source code **offline** (this is where we hit the main problem)

### 4.1 Why `pip download flash_attn-2.6.3.tar.gz` alone may NOT be enough
We downloaded `flash_attn-2.6.3.tar.gz` from Artifactory. It contained `hopper/setup.py` and a few Python files, but the build failed with:

```
ninja: error: .../hopper/flash_api.cpp ... missing and no known rule to make it
```

That happens when the **Hopper C++/CUDA sources are missing** from the source tree you’re building (e.g., `flash_api.cpp`, `flash_*_sm90*.cu`).

### 4.2 Check whether your local tree is a “full repo checkout” (GOOD) or a trimmed sdist (BAD)

#### If you already have a `flash-attention` repo directory:
```bash
cd ~/flash-attention/hopper

# These MUST exist for a source build:
ls -l flash_api.cpp
ls -1 flash_*sm90*.cu | head
```

#### If you only have the sdist tarball:
```bash
cd ~/fa3_src
tar -xzf flash_attn-2.6.3.tar.gz
cd flash_attn-2.6.3/hopper

# If this prints nothing, you cannot build FA3 from this sdist:
ls -l flash_api.cpp flash_*sm90*.cu 2>/dev/null || true
```

### 4.3 How to obtain a full source snapshot (recommended)
On an internet-connected machine (or internal git mirror), prepare a tarball and copy it to your server:

```bash
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.6.3
git submodule update --init --recursive
tar -czf flash-attention-v2.6.3-full.tar.gz flash-attention
```

Copy `flash-attention-v2.6.3-full.tar.gz` to the offline server (scp / internal artifact store), then:

```bash
cd ~
tar -xzf flash-attention-v2.6.3-full.tar.gz
```

Upstream install instructions assume you have this repo layout and install from `hopper/`.  
Source: https://github.com/Dao-AILab/flash-attention

---

## 5) Build + install FA3 (Hopper) on the server

### 5.1 Go to the Hopper package dir
```bash
cd ~/flash-attention/hopper
```

### 5.2 Set build environment variables (stable defaults)

**Limit parallel compilation** to avoid RAM blowups during nvcc compilation:
```bash
export MAX_JOBS=4
# if you still hit mysterious nvcc/ninja failures, try: export MAX_JOBS=1
```

Compile only for H100 (SM90):
```bash
export TORCH_CUDA_ARCH_LIST="9.0"
```

Ensure CUDA toolkit env is set:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
nvcc -V
```

### 5.3 Install build dependencies from your internal index
```bash
python -m pip install -U pip setuptools wheel
python -m pip install -U ninja packaging psutil einops pytest
```

### 5.4 Build/install (upstream method)
```bash
python setup.py install
```

Upstream explicitly documents:
```bash
cd hopper
python setup.py install
```
Source: https://github.com/Dao-AILab/flash-attention

---

## 6) Verify installation (the exact checks that finally worked)

### 6.1 Pin to an H100 (UUID) before verifying
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-83e5dd57-b016-f03a-03a5-50a385b1dcb6   # your H100 UUID
```

### 6.2 Verify installed modules and their file locations
```bash
python - <<'PY'
import importlib.util, torch

print("flash_attn_interface:", bool(importlib.util.find_spec("flash_attn_interface")))
print("flashattn_hopper_cuda:", bool(importlib.util.find_spec("flashattn_hopper_cuda")))

import flash_attn_interface, flashattn_hopper_cuda
print("flash_attn_interface file:", flash_attn_interface.__file__)
print("flashattn_hopper_cuda file:", flashattn_hopper_cuda.__file__)

print("GPU:", torch.cuda.get_device_name(0), "CC:", torch.cuda.get_device_capability(0))
print("CUDA_VISIBLE_DEVICES:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
PY
```

**Expected**:
- both specs are `True`
- GPU is H100, CC is `(9, 0)`
- `.so` is inside your venv `site-packages`

### 6.3 Smoke test: run a tiny attention call on H100
```bash
python - <<'PY'
import torch
from flash_attn_interface import flash_attn_func

assert torch.cuda.get_device_capability(0) == (9,0), "Not on H100 (sm90)!"

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, causal=True)
print("OK:", out.shape, out.dtype, out.device)
PY
```

Upstream explicitly indicates the FA3 import path is `flash_attn_interface` and exposes `flash_attn_func()`.  
Source: https://github.com/Dao-AILab/flash-attention

> Note: many people mistakenly check for a module named `flash_attn_3`. That’s not the import path used by the Hopper package as installed here; the correct entrypoint is `flash_attn_interface`. (This mismatch shows up in downstream issues, e.g., Transformers.)  
> Example: https://github.com/huggingface/transformers/issues/41179

---

## 7) Run the upstream test (optional but recommended)

From `~/flash-attention/hopper`:

```bash
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py
```

Source: https://github.com/Dao-AILab/flash-attention

Some environments may need a small test path patch; SURF documents one such fix:
https://servicedesk.surf.nl/wiki/pages/viewpage.action?pageId=187826305

---

## 8) Troubleshooting (the exact issues we hit)

### A) `nvcc: command not found`
- `nvcc` existed at `/usr/local/cuda/bin/nvcc`, but wasn’t on PATH.
- Fix:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
nvcc -V
```

### B) “not on H100” / you see `GPU: NVIDIA L40S`
- GPU 0 on your host is L40S, but FA3 is Hopper-only.
- Fix: pin to an H100 by UUID:
```bash
export CUDA_VISIBLE_DEVICES=GPU-<H100-UUID>
```
NVIDIA recommends UUIDs to avoid ambiguity.  
Sources:
- https://docs.nvidia.com/deploy/topics/topic_5_2_1.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html

### C) `ninja: error: .../hopper/flash_api.cpp ... missing`
- Your source tree does not include FA3 C++/CUDA sources; this commonly happens with a trimmed sdist.
- Fix: build from a full `flash-attention` repo snapshot (`git clone --recursive ...`) or install a prebuilt wheel via an internal artifact mirror.

### D) `RuntimeError: FlashAttention-3 is only supported on CUDA 12.3 and above`
- Your nvcc is too old (or you are picking up the wrong toolkit in PATH).
- Fix: use CUDA toolkit >= 12.3 and verify with `nvcc -V`.  
Source: https://github.com/Dao-AILab/flash-attention/issues/1477

### E) Minor CUDA mismatch warning (e.g., nvcc 12.6 vs torch cu121)
- You can see a warning from PyTorch C++ extension tooling about a minor version mismatch; this is often non-fatal.  
Source discussion: https://discuss.pytorch.org/t/pythorch-version-for-cuda-12-2/195217

---

## 9) “One-shot” command block (repeatable install)

Replace `<H100_UUID>` and run from a clean shell:

```bash
source /path/to/unlearning-venv/bin/activate

# pick H100
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<H100_UUID>

# cuda toolkit
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
nvcc -V

# build controls
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="9.0"

# deps
python -m pip install -U pip setuptools wheel
python -m pip install -U ninja packaging psutil einops pytest

# install from full repo checkout
cd ~/flash-attention/hopper
python setup.py install

# verify
python - <<'PY'
import importlib.util, torch
print("GPU:", torch.cuda.get_device_name(0), "CC:", torch.cuda.get_device_capability(0))
print("flash_attn_interface:", bool(importlib.util.find_spec("flash_attn_interface")))
print("flashattn_hopper_cuda:", bool(importlib.util.find_spec("flashattn_hopper_cuda")))
from flash_attn_interface import flash_attn_func
q = torch.randn(2,128,8,64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2,128,8,64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2,128,8,64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q,k,v, causal=True)
print("smoke:", out.shape, out.dtype)
PY
```

---

## References (primary)
- FlashAttention repository (FA3 requirements, install + import + test): https://github.com/Dao-AILab/flash-attention  
- NVIDIA `CUDA_VISIBLE_DEVICES` UUID guidance: https://docs.nvidia.com/deploy/topics/topic_5_2_1.html  
- CUDA Programming Guide env var appendix (UUID formats / abbreviations): https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html  
- FA3 CUDA >= 12.3 check (issue): https://github.com/Dao-AILab/flash-attention/issues/1477  
- PyTorch forum note about minor CUDA mismatch warnings: https://discuss.pytorch.org/t/pythorch-version-for-cuda-12-2/195217  
- SURF cluster install notes for FA3: https://servicedesk.surf.nl/wiki/pages/viewpage.action?pageId=187826305
