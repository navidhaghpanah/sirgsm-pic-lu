import gc
import io
import os
import sys
import json
import socket
import tempfile
import argparse
import threading
import subprocess
import webbrowser
import multiprocessing
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Runtime environment safeguards (must be set before importing torch/transformers)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Determine base paths for both normal Python and PyInstaller portable builds
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    APP_DIR = Path(sys.executable).resolve().parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    APP_DIR = Path(__file__).resolve().parent


def _resolve_data_dir() -> Path:
    """Use APP_DIR for portable storage; fall back to user app-data if APP_DIR is read-only."""
    probe = APP_DIR / f".write_test_{os.getpid()}"
    try:
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        return APP_DIR
    except OSError:
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        fallback = base / "SirgsmPicLu"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DATA_DIR = _resolve_data_dir()


def _configure_stdio() -> None:
    """Prevent crashes when stdout/stderr are None (--noconsole) or non-UTF-8 (Windows cmd)."""
    log_handle = None
    if sys.stdout is None or sys.stderr is None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            log_handle = open(DATA_DIR / "sirgsm_pic_lu.log", "a", encoding="utf-8", buffering=1)
        except OSError:
            log_handle = io.StringIO()
        if sys.stdout is None:
            sys.stdout = log_handle
        if sys.stderr is None:
            sys.stderr = log_handle
    else:
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass


def _prepare_windows_dlls() -> None:
    """Ensure PyTorch native DLL directory is registered first on Windows frozen builds."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    candidates = [
        BUNDLE_DIR / "torch" / "lib",
        APP_DIR / "_internal" / "torch" / "lib",
        APP_DIR / "torch" / "lib",
    ]
    for dll_dir in candidates:
        if dll_dir.is_dir():
            try:
                os.add_dll_directory(str(dll_dir))
            except OSError:
                pass


_configure_stdio()
_prepare_windows_dlls()

# Import torch BEFORE numpy and Pillow to avoid OpenMP/CRT DLL conflicts on Windows (WinError 1114)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file as load_safetensors  # noqa: E402
from transformers.models.clip.image_processing_clip import CLIPImageProcessor  # noqa: E402
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection  # noqa: E402
from transformers.models.distilbert.modeling_distilbert import DistilBertModel  # noqa: E402
from transformers.models.distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast  # noqa: E402

# Allow loading slightly truncated photos (common in messaging app downloads)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
APP_TITLE = "Sirgsm Pic Lu — AI Local Image Search"
IMAGE_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
IMAGE_MODEL_SUBFOLDER = "0_CLIPModel"
TEXT_MODEL_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
EMBEDDING_DIM = 512

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SKIP_DIR_NAMES = {
    "$recycle.bin",
    "system volume information",
    ".git",
    ".svn",
    "node_modules",
    "__pycache__",
    ".thumbnails",
    ".trash",
    ".trashes",
}

INDEX_CACHE_FILE = DATA_DIR / "image_index_cache.npz"
META_CACHE_FILE = DATA_DIR / "image_meta_cache.json"

MAX_PREPROCESS_SIDE = 512
THUMBNAIL_MAX_SIDE = 420
THUMBNAIL_CACHE_LIMIT = 256
QUERY_CACHE_LIMIT = 32

# Auto-open browser coordination (set only when launched as __main__)
_startup_open_url: Optional[str] = None


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_model_cache_dir() -> Optional[str]:
    """Use SIRGSM_MODEL_DIR or portable './models' folder next to the exe if present."""
    env_dir = os.environ.get("SIRGSM_MODEL_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    portable_models = APP_DIR / "models"
    if portable_models.is_dir():
        return str(portable_models)
    return None


def _strip_win_extended_prefix(path_str: str) -> str:
    if sys.platform == "win32":
        if path_str.startswith("\\\\?\\UNC\\"):
            return "\\\\" + path_str[8:]
        if path_str.startswith("\\\\?\\"):
            return path_str[4:]
    return path_str


def canonical_path(raw_path: str) -> str:
    """Resolve symlinks/relative parts and strip Windows extended prefix for clean display."""
    cleaned = raw_path.strip().strip('"').strip("'")
    resolved = os.path.abspath(os.path.realpath(os.path.expanduser(cleaned)))
    return _strip_win_extended_prefix(resolved)


def path_key(canon_path: str) -> str:
    """Platform-appropriate normalized key (case-insensitive on Windows)."""
    return os.path.normcase(canon_path)


def _file_signature(file_path: str) -> Optional[Tuple[int, int]]:
    """Return (size, mtime_ns) for incremental cache validation, or None if inaccessible."""
    try:
        st = os.stat(file_path)
        return int(st.st_size), int(st.st_mtime_ns)
    except OSError:
        return None


def _load_with_offline_fallback(loader_fn, model_label: str):
    """Load from local HuggingFace cache first (instant & offline); download only if missing."""
    try:
        return loader_fn(local_files_only=True)
    except Exception:
        pass
    try:
        print(f"Downloading / loading {model_label} (first run only)...")
        return loader_fn(local_files_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"خطا در بارگذاری یا دانلود مدل هوش مصنوعی ({model_label}). "
            "در اولین اجرا اتصال اینترنت (و در صورت محدودیت دسترسی به HuggingFace، ابزار تغییر IP یا تنظیم متغیر HF_ENDPOINT) لازم است. "
            "پس از اولین دانلود، برنامه کاملاً آفلاین کار می‌کند. "
            f"[جزئیات: {exc}]"
        ) from exc


class CLIPImageEncoder:
    """Memory-efficient CLIP ViT-B-32 vision encoder (exact same embeddings, no unused text tower)."""

    def __init__(self, device: torch.device, cache_dir: Optional[str] = None) -> None:
        self.device = device
        self.processor: CLIPImageProcessor = _load_with_offline_fallback(
            lambda local_files_only: CLIPImageProcessor.from_pretrained(
                IMAGE_MODEL_ID,
                subfolder=IMAGE_MODEL_SUBFOLDER,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ),
            "CLIP-ViT-B-32 Processor",
        )
        self.model: CLIPVisionModelWithProjection = _load_with_offline_fallback(
            lambda local_files_only: CLIPVisionModelWithProjection.from_pretrained(
                IMAGE_MODEL_ID,
                subfolder=IMAGE_MODEL_SUBFOLDER,
                cache_dir=cache_dir,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            ),
            "CLIP-ViT-B-32 Vision",
        )
        self.model.eval()
        self.model.to(self.device)

    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
            embeds = F.normalize(outputs.image_embeds.float(), p=2, dim=-1)
            result = embeds.cpu().numpy().astype(np.float32, copy=False)
        del inputs, pixel_values, outputs, embeds
        return result


class MultilingualCLIPTextEncoder:
    """Multilingual CLIP text encoder (Persian, English & 50+ languages) matching clip-ViT-B-32 space."""

    def __init__(self, device: torch.device, cache_dir: Optional[str] = None) -> None:
        self.device = device
        self.max_seq_length = 128
        self.tokenizer: DistilBertTokenizerFast = _load_with_offline_fallback(
            lambda local_files_only: DistilBertTokenizerFast.from_pretrained(
                TEXT_MODEL_ID,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ),
            "Multilingual-CLIP Tokenizer",
        )
        self.transformer: DistilBertModel = _load_with_offline_fallback(
            lambda local_files_only: DistilBertModel.from_pretrained(
                TEXT_MODEL_ID,
                cache_dir=cache_dir,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            ),
            "Multilingual-CLIP Text",
        )
        dense_path: str = _load_with_offline_fallback(
            lambda local_files_only: hf_hub_download(
                repo_id=TEXT_MODEL_ID,
                filename="2_Dense/model.safetensors",
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ),
            "Multilingual-CLIP Projection",
        )
        dense_weights = load_safetensors(dense_path, device="cpu")
        weight = dense_weights["linear.weight"]
        out_features, in_features = weight.shape
        self.dense = torch.nn.Linear(in_features, out_features, bias=False)
        self.dense.load_state_dict({"weight": weight})

        self.transformer.eval().to(self.device)
        self.dense.eval().to(self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            token_embeddings = self.transformer(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            projected = self.dense(pooled)
            normalized = F.normalize(projected.float(), p=2, dim=-1)
            return normalized.cpu().numpy().astype(np.float32, copy=False)


# Thread-safe lazy models (loaded independently to minimize RAM footprint)
_model_lock = threading.Lock()
_img_model: Optional[CLIPImageEncoder] = None
_text_model: Optional[MultilingualCLIPTextEncoder] = None

# Thread-safe in-memory search index
_index_lock = threading.RLock()
_image_items: List[Dict[str, Any]] = []  # [{"path": str, "size": int, "mtime_ns": int}]
_allowed_paths: Dict[str, Dict[str, Any]] = {}  # path_key -> item dict (O(1) security check)
_image_embeddings: Optional[np.ndarray] = None  # float32 [N, 512], L2-normalized

# Background indexing state
_job_lock = threading.Lock()
_cancel_event = threading.Event()
_index_job: Dict[str, Any] = {
    "running": False,
    "phase": "idle",  # idle | scanning | loading_model | encoding | saving | done | error | cancelled
    "processed": 0,
    "total": 0,
    "reused_count": 0,
    "new_count": 0,
    "failed_count": 0,
    "percent": 0,
    "message": "",
    "error": None,
}

# Small bounded LRU caches for fast thumbnails and repeated search queries
_thumb_lock = threading.Lock()
_thumb_cache: "OrderedDict[Tuple[str, int], bytes]" = OrderedDict()

_query_lock = threading.Lock()
_query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()


def get_image_model() -> CLIPImageEncoder:
    global _img_model
    with _model_lock:
        if _img_model is None:
            print("Loading CLIP vision model (clip-ViT-B-32)...")
            _img_model = CLIPImageEncoder(device=_get_device(), cache_dir=_get_model_cache_dir())
        return _img_model


def release_image_model() -> None:
    """Free vision encoder memory once indexing completes so idle/search RAM stays minimal."""
    global _img_model
    with _model_lock:
        if _img_model is not None:
            _img_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_text_model() -> MultilingualCLIPTextEncoder:
    global _text_model
    with _model_lock:
        if _text_model is None:
            print("Loading multilingual CLIP text model (Persian & English)...")
            _text_model = MultilingualCLIPTextEncoder(device=_get_device(), cache_dir=_get_model_cache_dir())
        return _text_model


def _rebuild_allowed_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {path_key(item["path"]): item for item in items}


def _set_index(items: List[Dict[str, Any]], embeddings: Optional[np.ndarray]) -> None:
    global _image_items, _allowed_paths, _image_embeddings
    with _index_lock:
        _image_items = items
        _allowed_paths = _rebuild_allowed_map(items)
        _image_embeddings = embeddings


def load_cache() -> None:
    """Load and validate cached embeddings (supports both legacy v1 list and v2 metadata formats)."""
    if not (INDEX_CACHE_FILE.exists() and META_CACHE_FILE.exists()):
        return
    try:
        raw_meta = json.loads(META_CACHE_FILE.read_text(encoding="utf-8"))
        items: List[Dict[str, Any]] = []

        if isinstance(raw_meta, list):
            # Backward compatibility with legacy v1 format: ["path1", "path2", ...]
            for p in raw_meta:
                if isinstance(p, str) and p.strip():
                    cp = canonical_path(p)
                    sig = _file_signature(cp)
                    items.append({
                        "path": cp,
                        "size": sig[0] if sig else -1,
                        "mtime_ns": sig[1] if sig else -1,
                    })
        elif isinstance(raw_meta, dict) and isinstance(raw_meta.get("items"), list):
            for entry in raw_meta["items"]:
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    items.append({
                        "path": canonical_path(entry["path"]),
                        "size": int(entry.get("size", -1)),
                        "mtime_ns": int(entry.get("mtime_ns", -1)),
                    })
        else:
            print("Cache metadata has unrecognized format; ignoring.")
            return

        with np.load(INDEX_CACHE_FILE, allow_pickle=False) as data:
            if "embeddings" not in data:
                print("Cache archive missing 'embeddings' array; ignoring.")
                return
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)

        if (
            len(items) == 0
            or embeddings.ndim != 2
            or embeddings.shape[0] != len(items)
            or embeddings.shape[1] != EMBEDDING_DIM
            or not np.isfinite(embeddings).all()
        ):
            print("Cache dimensions or values are invalid/out-of-sync; ignoring.")
            return

        # Ensure strict L2 normalization for accurate cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = (embeddings / norms).astype(np.float32, copy=False)

        _set_index(items, embeddings)
        print(f"Loaded {len(items)} cached images from {DATA_DIR}.")
    except Exception as exc:
        print(f"Failed to load cache (will rebuild on next index): {exc}")


def save_cache(items: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
    """Atomically write cache files so crashes or forced exits never corrupt saved indexes."""
    if embeddings is None or len(items) == 0:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_npz_path: Optional[Path] = None
    tmp_json_path: Optional[Path] = None
    try:
        fd_npz, raw_npz = tempfile.mkstemp(prefix=".idx_", suffix=".npz", dir=str(DATA_DIR))
        os.close(fd_npz)
        tmp_npz_path = Path(raw_npz)
        with open(tmp_npz_path, "wb") as f_npz:
            np.savez_compressed(f_npz, embeddings=embeddings.astype(np.float32, copy=False))

        fd_json, raw_json = tempfile.mkstemp(prefix=".meta_", suffix=".json", dir=str(DATA_DIR))
        tmp_json_path = Path(raw_json)
        payload = {"version": 2, "dim": EMBEDDING_DIM, "items": items}
        with os.fdopen(fd_json, "w", encoding="utf-8") as f_json:
            json.dump(payload, f_json, ensure_ascii=False)

        os.replace(tmp_npz_path, INDEX_CACHE_FILE)
        tmp_npz_path = None
        os.replace(tmp_json_path, META_CACHE_FILE)
        tmp_json_path = None
    finally:
        for leftover in (tmp_npz_path, tmp_json_path):
            if leftover is not None:
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass


def _open_image_for_clip(file_path: str) -> Image.Image:
    """Decode and downscale a single image safely with low peak memory and proper EXIF/alpha handling."""
    with Image.open(file_path) as raw:
        # Hint JPEG decoder to downscale during DCT decode (4-8x faster and far less RAM)
        if raw.format == "JPEG":
            try:
                raw.draft("RGB", (MAX_PREPROCESS_SIDE, MAX_PREPROCESS_SIDE))
            except Exception:
                pass

        # Seek first frame for animated GIF/WebP
        if getattr(raw, "is_animated", False):
            try:
                raw.seek(0)
            except Exception:
                pass

        img = ImageOps.exif_transpose(raw)

        # Composite transparent images onto white instead of black for better CLIP accuracy
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            rgba.close()
            if img is not raw:
                img.close()
            rgb = bg
        else:
            rgb = img.convert("RGB")
            if img is not raw and img is not rgb:
                img.close()

        # Bound memory while preserving full resolution for CLIP's 224x224 shortest-edge crop
        w, h = rgb.size
        if w <= 0 or h <= 0:
            rgb.close()
            raise ValueError("Empty image dimensions")
        shortest = min(w, h)
        if shortest > MAX_PREPROCESS_SIDE:
            scale = MAX_PREPROCESS_SIDE / float(shortest)
            new_size = (max(224, int(round(w * scale))), max(224, int(round(h * scale))))
            resized = rgb.resize(new_size, Image.Resampling.BICUBIC)
            rgb.close()
            rgb = resized

        return rgb


def _scan_folder_images(folder: Path, cancel_event: threading.Event) -> List[Tuple[str, int, int]]:
    """Walk folder recursively and collect (canonical_path, size, mtime_ns), skipping system dirs and symlink loops."""
    seen_keys = set()
    visited_dirs = set()
    discovered: List[Tuple[str, int, int]] = []

    for root, dirs, files in os.walk(folder, followlinks=True):
        if cancel_event.is_set():
            return []
        try:
            real_root = canonical_path(root)
            dir_key = path_key(real_root)
            if dir_key in visited_dirs:
                dirs[:] = []
                continue
            visited_dirs.add(dir_key)
        except OSError:
            dirs[:] = []
            continue

        # Prune hidden/system directories in-place
        dirs[:] = [
            d for d in dirs
            if d.lower() not in SKIP_DIR_NAMES and not (d.startswith(".") and len(d) > 1)
        ]
        dirs.sort()
        files.sort()
        for fname in files:
            if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            full_path = canonical_path(os.path.join(root, fname))
            key = path_key(full_path)
            if key in seen_keys:
                continue
            sig = _file_signature(full_path)
            if sig is None or sig[0] <= 0:
                continue
            seen_keys.add(key)
            discovered.append((full_path, sig[0], sig[1]))

    return discovered


def _update_job(**kwargs: Any) -> None:
    with _job_lock:
        _index_job.update(kwargs)


def _run_index_worker(folder: Path, append: bool, force_reindex: bool) -> None:
    """Background indexing worker: streams small batches with incremental cache reuse and low RAM."""
    try:
        _update_job(
            phase="scanning",
            processed=0,
            total=0,
            reused_count=0,
            new_count=0,
            failed_count=0,
            percent=0,
            message="در حال اسکن پوشه و یافتن عکس‌ها...",
            error=None,
        )

        discovered = _scan_folder_images(folder, _cancel_event)
        if _cancel_event.is_set():
            _update_job(running=False, phase="cancelled", message="عملیات ایندکس توسط کاربر لغو شد.")
            return

        if not discovered:
            _update_job(
                running=False,
                phase="error",
                error="هیچ عکسی با فرمت پشتیبانی‌شده در این پوشه پیدا نشد.",
                message="هیچ عکسی در این پوشه پیدا نشد.",
            )
            return

        # Snapshot current cache for O(1) incremental reuse of unchanged files
        with _index_lock:
            old_items = list(_image_items)
            old_embs = _image_embeddings

        existing_lookup: Dict[str, Tuple[Dict[str, Any], np.ndarray]] = {}
        if old_embs is not None and len(old_items) == old_embs.shape[0]:
            for idx, item in enumerate(old_items):
                existing_lookup[path_key(item["path"])] = (item, old_embs[idx])

        reused_items: List[Dict[str, Any]] = []
        reused_vecs: List[np.ndarray] = []
        to_encode: List[Tuple[str, int, int]] = []
        scanned_keys = set()

        for cpath, size, mtime_ns in discovered:
            key = path_key(cpath)
            scanned_keys.add(key)
            cached = None if force_reindex else existing_lookup.get(key)
            if (
                cached is not None
                and cached[0].get("size") == size
                and cached[0].get("mtime_ns") == mtime_ns
                and size > 0
            ):
                reused_items.append({"path": cpath, "size": size, "mtime_ns": mtime_ns})
                reused_vecs.append(cached[1])
            else:
                to_encode.append((cpath, size, mtime_ns))

        # When append=True, keep valid previously indexed images from other folders as well
        kept_other_items: List[Dict[str, Any]] = []
        kept_other_vecs: List[np.ndarray] = []
        if append and existing_lookup:
            for key, (item, vec) in existing_lookup.items():
                if key not in scanned_keys and os.path.isfile(item["path"]):
                    kept_other_items.append(item)
                    kept_other_vecs.append(vec)

        total_files = len(discovered)
        reused_count = len(reused_items)
        new_items: List[Dict[str, Any]] = []
        new_emb_chunks: List[np.ndarray] = []
        failed_count = 0

        _update_job(
            total=total_files,
            processed=reused_count,
            reused_count=reused_count,
            new_count=0,
            failed_count=0,
            percent=int((reused_count / max(total_files, 1)) * 100),
        )

        if to_encode:
            _update_job(
                phase="loading_model",
                message="در حال بارگذاری مدل هوش مصنوعی تصویر (بار اول ممکن است چند دقیقه طول بکشد)...",
            )
            img_model = get_image_model()

            if _cancel_event.is_set():
                _update_job(running=False, phase="cancelled", message="عملیات ایندکس توسط کاربر لغو شد.")
                return

            batch_size = 32 if img_model.device.type == "cuda" else 16
            processed_so_far = reused_count

            _update_job(
                phase="encoding",
                message=f"در حال پردازش {len(to_encode)} عکس جدید ({reused_count} عکس از کش خوانده شد)...",
            )

            batch_imgs: List[Image.Image] = []
            batch_meta: List[Dict[str, Any]] = []
            batches_done = 0

            def flush_batch() -> None:
                nonlocal batches_done, failed_count
                if not batch_imgs:
                    return
                try:
                    try:
                        chunk = img_model.encode_batch(batch_imgs)
                        new_emb_chunks.append(chunk)
                        new_items.extend(batch_meta)
                    except Exception:
                        # Isolate any problematic image instead of aborting the whole folder
                        for single_img, single_meta in zip(batch_imgs, batch_meta):
                            try:
                                single_chunk = img_model.encode_batch([single_img])
                                new_emb_chunks.append(single_chunk)
                                new_items.append(single_meta)
                            except Exception:
                                failed_count += 1
                finally:
                    for im in batch_imgs:
                        try:
                            im.close()
                        except Exception:
                            pass
                    batch_imgs.clear()
                    batch_meta.clear()
                    batches_done += 1
                    if batches_done % 8 == 0:
                        gc.collect()

            try:
                for cpath, size, mtime_ns in to_encode:
                    if _cancel_event.is_set():
                        for im in batch_imgs:
                            try:
                                im.close()
                            except Exception:
                                pass
                        batch_imgs.clear()
                        _update_job(running=False, phase="cancelled", message="عملیات ایندکس توسط کاربر لغو شد.")
                        return

                    try:
                        img = _open_image_for_clip(cpath)
                        batch_imgs.append(img)
                        batch_meta.append({"path": cpath, "size": size, "mtime_ns": mtime_ns})
                    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
                        failed_count += 1
                    except Exception:
                        failed_count += 1

                    if len(batch_imgs) >= batch_size:
                        flush_batch()

                    processed_so_far += 1
                    pct = int((processed_so_far / max(total_files, 1)) * 100)
                    _update_job(
                        processed=processed_so_far,
                        new_count=len(new_items) + len(batch_imgs),
                        failed_count=failed_count,
                        percent=min(pct, 99),
                        message=f"در حال پردازش عکس‌ها: {processed_so_far} از {total_files} ({pct}%)",
                    )

                if batch_imgs:
                    flush_batch()
            finally:
                # Always release vision model after indexing to reclaim ~350 MB RAM
                release_image_model()

        # Combine preserved + reused + newly encoded embeddings
        final_items: List[Dict[str, Any]] = []
        matrix_parts: List[np.ndarray] = []

        if kept_other_items:
            final_items.extend(kept_other_items)
            matrix_parts.append(np.vstack(kept_other_vecs).astype(np.float32, copy=False))
        if reused_items:
            final_items.extend(reused_items)
            matrix_parts.append(np.vstack(reused_vecs).astype(np.float32, copy=False))
        if new_items and new_emb_chunks:
            final_items.extend(new_items)
            matrix_parts.append(np.vstack(new_emb_chunks).astype(np.float32, copy=False))

        if not final_items or not matrix_parts:
            _update_job(
                running=False,
                phase="error",
                error="امکان خواندن عکس‌های این پوشه وجود ندارد (فایل‌ها خراب یا غیرقابل پشتیبانی هستند).",
                message="خطا در خواندن عکس‌ها.",
            )
            return

        final_embeddings = np.ascontiguousarray(np.vstack(matrix_parts), dtype=np.float32)

        _update_job(phase="saving", percent=99, message="در حال ذخیره فایل کش...")
        _set_index(final_items, final_embeddings)
        save_cache(final_items, final_embeddings)
        gc.collect()

        details = []
        if new_items:
            details.append(f"{len(new_items)} جدید")
        if reused_count:
            details.append(f"{reused_count} بدون تغییر از کش")
        if failed_count:
            details.append(f"{failed_count} ناموفق")
        detail_str = f" ({'، '.join(details)})" if details else ""

        _update_job(
            running=False,
            phase="done",
            processed=total_files,
            total=total_files,
            reused_count=reused_count,
            new_count=len(new_items),
            failed_count=failed_count,
            percent=100,
            error=None,
            message=f"{len(final_items)} عکس آماده جستجو است{detail_str}.",
        )
    except Exception as exc:
        release_image_model()
        err_msg = str(exc) or "خطای غیرمنتظره هنگام ایندکس پوشه."
        _update_job(
            running=False,
            phase="error",
            error=err_msg,
            message=err_msg,
        )


def _encode_query_cached(query: str) -> np.ndarray:
    normalized_q = " ".join(query.strip().split())
    with _query_lock:
        cached = _query_cache.get(normalized_q)
        if cached is not None:
            _query_cache.move_to_end(normalized_q)
            return cached

    text_model = get_text_model()
    vec = text_model.encode([normalized_q])[0]

    with _query_lock:
        _query_cache[normalized_q] = vec
        if len(_query_cache) > QUERY_CACHE_LIMIT:
            _query_cache.popitem(last=False)
    return vec


def _get_allowed_file(raw_path: str) -> Tuple[Path, Dict[str, Any]]:
    """Validate requested image path against indexed allowlist in O(1) time."""
    cpath = canonical_path(raw_path)
    key = path_key(cpath)
    with _index_lock:
        item = _allowed_paths.get(key)
    if item is None:
        raise HTTPException(status_code=403, detail="دسترسی به این فایل مجاز نیست.")
    file_path = Path(item["path"])
    if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="فایل پیدا نشد (ممکن است حذف یا جابه‌جا شده باشد).")
    return file_path, item


def _make_thumbnail_bytes(file_path: Path, mtime_ns: int) -> bytes:
    cache_key = (path_key(str(file_path)), mtime_ns)
    with _thumb_lock:
        hit = _thumb_cache.get(cache_key)
        if hit is not None:
            _thumb_cache.move_to_end(cache_key)
            return hit

    with Image.open(file_path) as raw:
        if raw.format == "JPEG":
            try:
                raw.draft("RGB", (THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE))
            except Exception:
                pass
        if getattr(raw, "is_animated", False):
            try:
                raw.seek(0)
            except Exception:
                pass
        img = ImageOps.exif_transpose(raw)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (15, 23, 42))
            bg.paste(rgba, mask=rgba.split()[3])
            rgba.close()
            if img is not raw:
                img.close()
            rgb = bg
        else:
            rgb = img.convert("RGB")
            if img is not raw and img is not rgb:
                img.close()

        rgb.thumbnail((THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=82, optimize=True)
        rgb.close()
        data = buf.getvalue()

    with _thumb_lock:
        _thumb_cache[cache_key] = data
        while len(_thumb_cache) > THUMBNAIL_CACHE_LIMIT:
            _thumb_cache.popitem(last=False)
    return data


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_cache()
    if _startup_open_url:
        # Open browser only after the server is actually bound and ready
        threading.Timer(0.4, lambda: webbrowser.open(_startup_open_url)).start()
    yield


app = FastAPI(title=APP_TITLE, lifespan=lifespan)


class IndexRequest(BaseModel):
    folder_path: str
    append: bool = False
    force_reindex: bool = False


class PathRequest(BaseModel):
    path: str


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = BUNDLE_DIR / "static" / "index.html"
    if not html_path.is_file():
        raise HTTPException(status_code=500, detail="فایل رابط کاربری (static/index.html) پیدا نشد.")
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    with _index_lock:
        count = len(_image_items)
        ready = _image_embeddings is not None and count > 0
    with _job_lock:
        job_snapshot = dict(_index_job)
    return {
        "indexed_count": count,
        "ready": ready,
        "device": str(_get_device()),
        "indexing": job_snapshot,
    }


@app.post("/api/index")
def index_folder(req: IndexRequest):
    cleaned = req.folder_path.strip().strip('"').strip("'")
    if not cleaned:
        raise HTTPException(status_code=400, detail="لطفاً مسیر پوشه عکس‌ها را وارد کنید.")

    try:
        folder = Path(canonical_path(cleaned))
    except Exception:
        raise HTTPException(status_code=400, detail="فرمت مسیر پوشه معتبر نیست.")

    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="مسیر پوشه معتبر نیست یا وجود ندارد.")

    with _job_lock:
        if _index_job["running"]:
            raise HTTPException(status_code=409, detail="یک عملیات ایندکس در حال انجام است. لطفاً صبر کنید یا آن را لغو کنید.")
        _cancel_event.clear()
        _index_job.update({
            "running": True,
            "phase": "scanning",
            "processed": 0,
            "total": 0,
            "reused_count": 0,
            "new_count": 0,
            "failed_count": 0,
            "percent": 0,
            "message": "شروع اسکن پوشه...",
            "error": None,
        })

    worker = threading.Thread(
        target=_run_index_worker,
        args=(folder, req.append, req.force_reindex),
        daemon=True,
        name="sirgsm-indexer",
    )
    worker.start()

    return {
        "started": True,
        "folder": str(folder),
        "message": "عملیات ایندکس در پس‌زمینه آغاز شد.",
    }


@app.post("/api/index/cancel")
def cancel_index():
    with _job_lock:
        if not _index_job["running"]:
            return {"cancelled": False, "message": "عملیات فعالی در حال اجرا نیست."}
        _cancel_event.set()
        _index_job["message"] = "در حال توقف عملیات ایندکس..."
    return {"cancelled": True, "message": "درخواست لغو ثبت شد."}


@app.get("/api/search")
def search_images(
    q: str = Query(..., min_length=1, max_length=300),
    top_k: int = Query(24, ge=1, le=120),
):
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="عبارت جستجو نمی‌تواند خالی باشد.")

    with _index_lock:
        items = _image_items
        embeddings = _image_embeddings

    if embeddings is None or len(items) == 0:
        raise HTTPException(status_code=400, detail="ابتدا یک پوشه عکس را ایندکس کنید.")

    try:
        text_emb = _encode_query_cached(query)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Fast cosine similarity (both image and text embeddings are L2-normalized float32)
    scores = embeddings @ text_emb
    scores = np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)

    # O(N) top-k selection using argpartition instead of sorting the entire index
    k = min(top_k, len(scores))
    if k < len(scores):
        candidate_idx = np.argpartition(scores, -k)[-k:]
        top_indices = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    else:
        top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        item = items[int(idx)]
        score_pct = round(max(0.0, min(100.0, float(scores[int(idx)]) * 100.0)), 1)
        results.append({
            "path": item["path"],
            "filename": os.path.basename(item["path"]),
            "score": score_pct,
            "v": int(item.get("mtime_ns", 0)) // 1_000_000,
        })

    return {"query": query, "total_indexed": len(items), "results": results}


@app.get("/api/thumbnail")
def get_thumbnail(path: str):
    file_path, item = _get_allowed_file(path)
    sig = _file_signature(str(file_path))
    mtime_ns = sig[1] if sig else int(item.get("mtime_ns", 0))
    try:
        jpeg_bytes = _make_thumbnail_bytes(file_path, mtime_ns)
        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    except Exception:
        return FileResponse(file_path, headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/image")
def get_image(path: str):
    file_path, _ = _get_allowed_file(path)
    return FileResponse(file_path, headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/pick-folder")
def pick_folder():
    """Open native OS folder picker dialog (uses PowerShell on Windows so tkinter isn't needed in EXE)."""
    try:
        if sys.platform == "win32":
            ps_script = (
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$dlg.Description = 'Select your photos folder - Sirgsm Pic Lu'; "
                "$dlg.ShowNewFolderButton = $false; "
                "$top = New-Object System.Windows.Forms.Form; "
                "$top.TopMost = $true; "
                "if ($dlg.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { "
                "Write-Output $dlg.SelectedPath }; "
                "$top.Dispose(); $dlg.Dispose()"
            )
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                creationflags=creationflags,
            )
            selected = proc.stdout.strip()
            return {"folder_path": selected or None}

        if sys.platform == "darwin":
            proc = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select photos folder")'],
                capture_output=True,
                text=True,
                timeout=180,
            )
            selected = proc.stdout.strip() if proc.returncode == 0 else ""
            return {"folder_path": selected or None}

        # Linux desktop (zenity / kdialog if available)
        for cmd in (
            ["zenity", "--file-selection", "--directory", "--title=Select photos folder"],
            ["kdialog", "--getexistingdirectory", str(Path.home())],
        ):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if proc.returncode == 0 and proc.stdout.strip():
                    return {"folder_path": proc.stdout.strip()}
                if proc.returncode != 0:
                    return {"folder_path": None}
            except FileNotFoundError:
                continue
    except subprocess.TimeoutExpired:
        return {"folder_path": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"خطا در باز کردن پنجره انتخاب پوشه: {exc}")

    raise HTTPException(
        status_code=501,
        detail="انتخابگر گرافیکی پوشه در این سیستم فعال نیست؛ لطفاً مسیر پوشه را دستی وارد کنید.",
    )


@app.post("/api/reveal")
def reveal_in_folder(req: PathRequest):
    """Reveal an indexed image inside the OS file manager (safe list args, no shell)."""
    file_path, _ = _get_allowed_file(req.path)
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer.exe", f"/select,{file_path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(file_path)])
        else:
            subprocess.Popen(["xdg-open", str(file_path.parent)])
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"امکان باز کردن پوشه وجود ندارد: {exc}")


def find_free_port(host: str, preferred_port: int = 8000, max_tries: int = 20) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"هیچ پورت آزادی در بازه {preferred_port} تا {preferred_port + max_tries - 1} پیدا نشد.")


def run_self_test() -> int:
    """Offline smoke test used by build_portable.bat & GitHub Actions to verify the compiled EXE."""
    from transformers.models.clip.configuration_clip import CLIPVisionConfig
    from transformers.models.distilbert.configuration_distilbert import DistilBertConfig

    print(f"[self-test] Python {sys.version.split()[0]} | frozen={getattr(sys, 'frozen', False)}")
    print(f"[self-test] torch={torch.__version__} | numpy={np.__version__} | device={_get_device()}")
    print(f"[self-test] BUNDLE_DIR={BUNDLE_DIR} | DATA_DIR={DATA_DIR}")

    # 1. Static UI check
    html_file = BUNDLE_DIR / "static" / "index.html"
    assert html_file.is_file(), f"Missing bundled static file: {html_file}"
    html_text = html_file.read_text(encoding="utf-8")
    assert "Sirgsm Pic Lu" in html_text, "Unexpected index.html content"

    # 2. CLIP vision forward pass + Pillow preprocessing check (offline, tiny config)
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = Path(tmp) / "sample.jpg"
        Image.new("RGB", (800, 600), color=(20, 130, 220)).save(sample_path, format="JPEG")
        pre_img = _open_image_for_clip(str(sample_path))
        try:
            proc = CLIPImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224})
            vcfg = CLIPVisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                projection_dim=32,
                image_size=224,
                patch_size=32,
            )
            vmodel = CLIPVisionModelWithProjection(vcfg).eval()
            pixels = proc(images=[pre_img], return_tensors="pt")["pixel_values"]
            with torch.inference_mode():
                vout = F.normalize(vmodel(pixel_values=pixels).image_embeds, p=2, dim=-1).numpy()
            assert vout.shape == (1, 32) and np.isfinite(vout).all()
        finally:
            pre_img.close()

        # 3. DistilBERT + Dense + safetensors round-trip check (offline, tiny config)
        from safetensors.torch import save_file as save_safetensors

        dcfg = DistilBertConfig(vocab_size=256, dim=64, hidden_dim=128, n_layers=2, n_heads=4)
        dmodel = DistilBertModel(dcfg).eval()
        dense_file = Path(tmp) / "dense.safetensors"
        save_safetensors({"linear.weight": torch.randn(32, 64)}, str(dense_file))
        loaded_w = load_safetensors(str(dense_file), device="cpu")["linear.weight"]
        dense = torch.nn.Linear(64, 32, bias=False)
        dense.load_state_dict({"weight": loaded_w})
        dense.eval()

        input_ids = torch.tensor([[10, 25, 42, 12]], dtype=torch.long)
        attn_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            h = dmodel(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / attn_mask.sum(dim=1, keepdim=True)
            tout = F.normalize(dense(pooled), p=2, dim=-1).numpy()
        assert tout.shape == (1, 32) and np.isfinite(tout).all()

        # 4. Rust Fast Tokenizer check (Persian + English, offline)
        vocab_file = Path(tmp) / "vocab.txt"
        vocab_file.write_text(
            "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "دریا", "کوه", "sea"]) + "\n",
            encoding="utf-8",
        )
        fast_tok = DistilBertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)
        tok_out = fast_tok(["دریا sea"], padding=True, truncation=True, max_length=16, return_tensors="pt")
        assert tok_out["input_ids"].shape[0] == 1

    # 5. Uvicorn + FastAPI route smoke test
    import uvicorn  # noqa: F401
    import uvicorn.loops.auto  # noqa: F401
    import uvicorn.protocols.http.auto  # noqa: F401
    import uvicorn.lifespan.on  # noqa: F401

    assert status()["indexed_count"] >= 0
    assert len(home()) > 500
    print("[self-test] PASSED: All modules, DLLs, metadata, tokenizer, models, and routes verified.")
    return 0


def main() -> None:
    global _startup_open_url
    import uvicorn

    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Preferred port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--self-test", action="store_true", help="Run offline binary self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_test())

    port = find_free_port(args.host, preferred_port=args.port)
    browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browse_host}:{port}"

    if not args.no_browser:
        _startup_open_url = url

    print("=" * 56)
    print(f"  {APP_TITLE}")
    print(f"  Server URL : {url}")
    print(f"  Cache Dir  : {DATA_DIR}")
    print(f"  Device     : {_get_device()}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 56)

    # Safe log_config when running without a TTY (prevents isatty() AttributeError in frozen builds)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    # Required on Windows for PyInstaller executables to prevent infinite subprocess spawning
    multiprocessing.freeze_support()
    main()
