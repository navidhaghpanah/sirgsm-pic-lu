# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Sirgsm Pic Lu (Windows / Linux / macOS portable build).

Default mode : onedir  -> fast startup (1-2s), no %TEMP% unpacking, no 4 GB limit.
Single-file  : set SIRGSM_ONEFILE=1 before running pyinstaller -> single .exe output.
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

APP_NAME = "Sirgsm-Pic-Lu"
ONEFILE = os.environ.get("SIRGSM_ONEFILE", "0").strip().lower() in ("1", "true", "yes")
SPEC_DIR = Path(SPECPATH).resolve()

# Bundle static frontend assets
datas = [
    (str(SPEC_DIR / "static"), "static"),
]

# Packages queried via importlib.metadata at runtime by transformers / huggingface_hub / fastapi
METADATA_PACKAGES = [
    "torch",
    "transformers",
    "huggingface-hub",
    "hf-xet",
    "tokenizers",
    "safetensors",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "PyYAML",
    "fsspec",
    "certifi",
    "urllib3",
    "charset-normalizer",
    "idna",
    "Pillow",
    "fastapi",
    "starlette",
    "uvicorn",
    "pydantic",
    "pydantic_core",
    "anyio",
    "opentelemetry-api",
]

for pkg in METADATA_PACKAGES:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass


def _safe_submodules(pkg_name: str):
    try:
        return collect_submodules(pkg_name)
    except Exception:
        return []


# Collect ONLY the exact model architectures we need (avoids bundling 300+ unused LLM modules)
hiddenimports = [
    # Uvicorn dynamic imports
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # Transformers / Hub / Safetensors runtime imports
    "safetensors.torch",
    "tokenizers",
    "huggingface_hub",
    "hf_xet",
    "transformers.generation.utils",
]
hiddenimports += _safe_submodules("transformers.models.clip")
hiddenimports += _safe_submodules("transformers.models.distilbert")
hiddenimports += _safe_submodules("numpy._core")

# Strip heavy unused packages, test suites, and optional ML backends to minimize EXE size
excludes = [
    "scipy",
    "sklearn",
    "scikit-learn",
    "joblib",
    "pandas",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "_pytest",
    "tkinter",
    "_tkinter",
    "tcl",
    "tk",
    "turtle",
    "turtledemo",
    "pydoc_data",
    "lib2to3",
    "tensorboard",
    "tensorflow",
    "keras",
    "flax",
    "jax",
    "jaxlib",
    "onnx",
    "onnxruntime",
    "cv2",
    "torchvision",
    "torchaudio",
    "triton",
]

# Optional app icon if icon.ico is placed in the project root
icon_file = SPEC_DIR / "icon.ico"
exe_icon = str(icon_file) if icon_file.is_file() else None

a = Analysis(
    [str(SPEC_DIR / "app.py")],
    pathex=[str(SPEC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)


def _is_unneeded_torch_dev_file(dest_name: str) -> bool:
    """Drop C++ headers, CMake files, static link libs, and C++ test binaries (~150-220 MB saved)."""
    norm = dest_name.replace("\\", "/").lower()
    if norm.startswith(("torch/test/", "torch/include/", "torch/share/")):
        return True
    if norm.startswith("torch/bin/") and "torch_shm_manager" not in norm:
        return True
    if norm.startswith("torch/lib/") and norm.endswith((".lib", ".a", ".pdb")):
        return True
    return False


a.binaries = [entry for entry in a.binaries if not _is_unneeded_torch_dev_file(entry[0])]
a.datas = [entry for entry in a.datas if not _is_unneeded_torch_dev_file(entry[0])]

pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        # IMPORTANT: Disable UPX — compressing PyTorch DLLs causes WinError 1114 & AV false positives
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=exe_icon,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=exe_icon,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=APP_NAME,
    )
