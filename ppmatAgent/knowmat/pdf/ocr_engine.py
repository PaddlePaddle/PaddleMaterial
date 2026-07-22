"""OCR engine creation and execution helpers.

Optional ``OCR_INFER_TIMEOUT_SEC`` (>0) wraps each inference call with a wait
timeout. A timeout only stops waiting in the caller thread; native/CUDA work in
the worker thread may continue until completion (Python cannot hard-cancel it).
"""

from __future__ import annotations

import functools
import gc
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path
from statistics import median
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def log_performance(func: F) -> F:
    """Decorator to log execution time and GPU memory usage for OCR functions.
    
    This decorator wraps functions to provide performance metrics including:
    - Execution time in seconds
    - GPU memory usage before and after (if available)
    
    Parameters
    ----------
    func : Callable
        The function to decorate.
        
    Returns
    -------
    Callable
        The decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        
        # Log GPU memory before
        used_before, total = get_gpu_memory_info()
        if total > 0:
            logger.debug(
                "[%s] GPU memory before: %.1f/%.1f GB (%.1f%% used)",
                func.__name__, used_before, total, (used_before / total) * 100
            )
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Log GPU memory after
            used_after, total = get_gpu_memory_info()
            if total > 0:
                delta = used_after - used_before
                logger.info(
                    "[%s] completed in %.1fs, GPU: %.1f->%.1f GB (%+.2f GB)",
                    func.__name__, elapsed, used_before, used_after, delta
                )
            else:
                logger.info("[%s] completed in %.1fs", func.__name__, elapsed)
            
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("[%s] failed after %.1fs: %s", func.__name__, elapsed, e)
            raise
    return wrapper  # type: ignore


def default_model_dir() -> Path:
    """Return the default local model directory in this project."""
    repo_root = Path(__file__).resolve().parents[3]
    
    version = os.getenv("PADDLEOCRVL_VERSION", "1.5")
    default_subdir = "paddleocrvl1_0" if version == "1.0" else "paddleocrvl1_5"
    
    model_dir = os.getenv("PADDLEOCRVL_MODEL_DIR", str(repo_root / "models" / default_subdir))
    return Path(model_dir).expanduser().resolve()


def _prepare_ocr_home(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLEOCR_HOME"] = str(model_dir)
    if "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" not in os.environ:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def _paddle_place_looks_undefined(place_repr: str) -> bool:
    s = (place_repr or "").lower()
    return "undefined" in s


def _gpu_card_index(device: str) -> int:
    """Parse ``gpu:1`` / ``cuda:0`` style strings; default 0."""
    d = (device or "").strip().lower()
    if ":" in d:
        try:
            return int(d.rsplit(":", 1)[-1])
        except ValueError:
            return 0
    return 0


def _paddle_device_synchronize() -> None:
    """Prefer ``paddle.device.synchronize`` (Paddle >= 2.5); fall back to cuda API."""
    try:
        import paddle  # type: ignore

        sync = getattr(paddle.device, "synchronize", None)
        if callable(sync):
            sync()
        else:
            paddle.device.cuda.synchronize()
    except Exception:
        pass


def _warm_paddle_gpu_context(device: str) -> None:
    """Force a valid CUDA default context.

    After VL ``close()`` / ``empty_cache()``, ``paddle.get_device()`` may still
    report ``gpu:0`` while native code passes ``Place(undefined:0)`` into
    ``is_bfloat16_supported``. A tiny device tensor plus sync tends to realign
    the executor place with CUDA.
    """
    try:
        import paddle  # type: ignore
    except ImportError:
        return

    try:
        paddle.disable_static()
    except Exception:
        pass

    idx = _gpu_card_index(device)
    try:
        place = paddle.CUDAPlace(idx)  # type: ignore[attr-defined]
        try:
            paddle.device.set_device(place)
        except Exception:
            paddle.device.set_device(device)
    except Exception:
        try:
            paddle.device.set_device(device)
        except Exception:
            return

    _paddle_device_synchronize()

    try:
        x = paddle.zeros([1], dtype="float32")
        try:
            _ = float(x.item())
        except Exception:
            _ = x.numpy()
        del x
    except Exception as exc:
        logger.warning("_warm_paddle_gpu_context: device tensor probe failed: %s", exc)
        return

    _paddle_device_synchronize()


def ensure_paddle_device_from_env() -> None:
    """Set Paddle's default device before VL init.

    After ``engine.close()``, ``empty_cache()``, or similar, the active place can
    become ``Place(undefined:0)``, and ``is_bfloat16_supported`` then raises.
    Re-pinning from ``KNOWMAT_OCR_DEVICE`` (or ``gpu:0`` / ``cpu``) avoids that.
    """
    try:
        import paddle  # type: ignore
    except ImportError:
        return

    raw = (os.getenv("KNOWMAT_OCR_DEVICE") or "").strip()
    if raw:
        device = raw
    else:
        try:
            has_cuda = bool(paddle.device.is_compiled_with_cuda())
        except Exception:
            has_cuda = False
        device = "gpu:0" if has_cuda else "cpu"

    def _apply(dev: str) -> bool:
        try:
            paddle.device.set_device(dev)
            return True
        except Exception as e1:
            try:
                paddle.set_device(dev)  # type: ignore[attr-defined]
                return True
            except Exception as e2:
                logger.warning(
                    "ensure_paddle_device_from_env: could not set %r: %s; legacy set_device: %s",
                    dev,
                    e1,
                    e2,
                )
                return False

    if not _apply(device):
        logger.warning(
            "ensure_paddle_device_from_env: failed to pin device %r; "
            "PaddleOCR-VL init may hit Place(undefined:0). Check KNOWMAT_OCR_DEVICE / CUDA.",
            device,
        )
        return

    # Verify default place after cleanup paths (close/empty_cache) left it valid.
    try:
        cur = str(paddle.get_device())
    except Exception:
        cur = ""
    if _paddle_place_looks_undefined(cur):
        logger.warning(
            "Paddle default device is still %r after set_device(%r); retrying set_device once.",
            cur,
            device,
        )
        _apply(device)

    # String device can look valid while C++ still uses Place(undefined:0) until a GPU op runs.
    try:
        has_cuda = bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        has_cuda = False
    dl = device.lower()
    if has_cuda and (dl.startswith("gpu") or dl.startswith("cuda")):
        _warm_paddle_gpu_context(device)


def _installed_dist_version(name: str) -> Optional[str]:
    try:
        return dist_version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _validate_paddle_ocr_runtime() -> None:
    """Fail fast on broken Paddle CPU/GPU combinations."""
    try:
        import paddle  # type: ignore
    except ImportError:
        return

    requested = (os.getenv("KNOWMAT_OCR_DEVICE") or "").strip().lower()
    wants_gpu = requested.startswith(("gpu", "cuda"))

    try:
        has_cuda = bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        has_cuda = False

    cpu_pkg = _installed_dist_version("paddlepaddle")
    gpu_pkg = _installed_dist_version("paddlepaddle-gpu")

    if has_cuda:
        return

    if wants_gpu:
        raise RuntimeError(
            "KNOWMAT_OCR_DEVICE requests GPU OCR, but the imported Paddle runtime is CPU-only "
            f"(compiled_with_cuda=False). Installed packages: paddlepaddle={cpu_pkg or 'not installed'}, "
            f"paddlepaddle-gpu={gpu_pkg or 'not installed'}. "
            "Remove the CPU paddlepaddle package, reinstall paddlepaddle-gpu, and verify "
            "`python -c \"import paddle; print(paddle.device.is_compiled_with_cuda())\"` returns True."
        )

    if cpu_pkg and gpu_pkg:
        raise RuntimeError(
            "Detected both paddlepaddle=%s and paddlepaddle-gpu=%s, but the imported Paddle runtime "
            "reports compiled_with_cuda=False. This usually means the CPU package overwrote the GPU "
            "runtime. Remove `paddlepaddle`, reinstall `paddlepaddle-gpu`, and verify "
            "`python -c \"import paddle; print(paddle.device.is_compiled_with_cuda())\"` returns True."
            % (cpu_pkg, gpu_pkg)
        )
    elif gpu_pkg:
        logger.error(
            "paddlepaddle-gpu=%s is installed, but imported Paddle reports compiled_with_cuda=False. "
            "Check CUDA/cuDNN compatibility and reinstall the GPU runtime.",
            gpu_pkg,
        )
    else:
        raise RuntimeError(
            "Local OCR requires an NVIDIA GPU with paddlepaddle-gpu installed. "
            "CPU-only local OCR is no longer supported. "
            "Use PaddleOCR API for cloud OCR without local GPU setup, "
            "or install paddlepaddle-gpu for local PaddleOCR-VL inference. "
            "The MinerU backend is available only as a compatibility fallback."
        )


def _legacy_paddleocr_allowed() -> bool:
    """Explicit opt-in for legacy engine only (debug / broken VL installs)."""
    v = (os.getenv("KNOWMAT_ALLOW_LEGACY_PADDLEOCR") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def create_ocr_engine(model_dir: Path) -> Tuple[Any, str]:
    """Create a PaddleOCR-VL engine.

    By default KnowMat does **not** fall back to legacy ``PaddleOCR`` (which would
    skip PP-StructureV3 refinement). Set ``KNOWMAT_ALLOW_LEGACY_PADDLEOCR=1`` only
    for debugging or emergency use.

    Engine parameters are aligned with the StarRiver community pipeline
    (model_settings extracted from StarRiver OCR outputs):
      - use_layout_detection=True: enables PP-Structure layout detection
      - use_seal_recognition=True: detect seal/stamp regions
      - format_block_content=True: format block content for downstream use
      - merge_layout_blocks=True: merge adjacent blocks of same type

    PaddleOCR-VL 1.5 does not accept ``use_angle_cls`` (raises ValueError).
    """
    _prepare_ocr_home(model_dir)
    ensure_paddle_device_from_env()
    _validate_paddle_ocr_runtime()
    legacy_ok = _legacy_paddleocr_allowed()

    try:
        from paddleocr import PaddleOCRVL  # type: ignore
    except ImportError as exc:
        if legacy_ok:
            logger.warning(
                "PaddleOCRVL is unavailable; KNOWMAT_ALLOW_LEGACY_PADDLEOCR is set, so KnowMat will fall back to legacy PaddleOCR."
            )
        else:
            raise ImportError(
                "PaddleOCRVL is not available. Install OCR dependencies with `pip install -r ppmatAgent/requirements-optional.txt`, "
                "or GPU OCR with `pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu129/`. "
                "For temporary debugging only, set KNOWMAT_ALLOW_LEGACY_PADDLEOCR=1 to allow fallback."
            ) from exc
    else:
        version = (os.getenv("PADDLEOCRVL_VERSION", "1.5") or "1.5").strip()
        star_aligned_kwargs: dict = {
            "use_layout_detection": True,
            "use_seal_recognition": True,
            "format_block_content": True,
            "merge_layout_blocks": True,
        }
        candidates: List[dict] = []
        if version == "1.0":
            candidates.append({"pipeline_version": "v1", **star_aligned_kwargs})
            candidates.append({"pipeline_version": "v1"})
            candidates.append({})
        else:
            candidates.append({"pipeline_version": "v1.5", **star_aligned_kwargs})
            candidates.append({**star_aligned_kwargs})
            candidates.append({"pipeline_version": "v1.5"})
            candidates.append({})

        last_vl_error: Optional[BaseException] = None
        for kwargs in candidates:
            ensure_paddle_device_from_env()
            try:
                engine = PaddleOCRVL(**kwargs)
                logger.info(
                    "PaddleOCR-VL ready (PADDLEOCRVL_VERSION=%s, pipeline_version=%r).",
                    version,
                    kwargs.get("pipeline_version", "(default)"),
                )
                return engine, "paddleocrvl"
            except TypeError as exc:
                last_vl_error = exc
                logger.debug("PaddleOCRVL TypeError for kwargs %s: %s", kwargs, exc)
                continue
            except Exception as exc:
                last_vl_error = exc
                logger.warning(
                    "PaddleOCRVL init failed (will try next candidate): %s: %s",
                    type(exc).__name__,
                    exc,
                )
                continue

        if last_vl_error is not None:
            if legacy_ok:
                logger.error(
                    "All PaddleOCR-VL initialization candidates failed (%s: %s); falling back because KNOWMAT_ALLOW_LEGACY_PADDLEOCR is enabled.",
                    type(last_vl_error).__name__,
                    last_vl_error,
                )
            else:
                raise RuntimeError(
                    "PaddleOCR-VL initialization failed and KnowMat will not silently fall back to legacy PaddleOCR. "
                    f"Last error: {type(last_vl_error).__name__}: {last_vl_error}. "
                    "Try setting KNOWMAT_OCR_DEVICE=gpu:0 (or cpu), check your CUDA/Paddle install, and make sure GPU memory was released after the previous PDF. "
                    "For debugging only, set KNOWMAT_ALLOW_LEGACY_PADDLEOCR=1."
                ) from last_vl_error

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PaddleOCR is not installed. Install OCR dependencies with `pip install -r ppmatAgent/requirements-optional.txt`, "
            "or GPU OCR with `pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu129/`."
        ) from exc

    init_candidates = (
        {"use_angle_cls": True, "lang": "en"},
        {"lang": "en"},
        {},
    )
    last_error: Exception | None = None
    for kwargs in init_candidates:
        try:
            logger.warning(
                "Using legacy PaddleOCR engine (not PaddleOCR-VL); "
                "PP-StructureV3 will still contribute layout seeding and route_and_reocr refinement."
            )
            return PaddleOCR(**kwargs), "paddleocr"
        except TypeError as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(f"Failed to initialize PaddleOCR engine: {last_error}") from last_error
    raise RuntimeError("Failed to initialize PaddleOCR engine.")


def try_release_paddle_gpu_memory() -> None:
    """Best-effort: free Paddle GPU allocations between PDFs (reduces OOM on 8GB cards)."""
    gc.collect()
    try:
        import paddle  # type: ignore

        is_cuda = False
        if hasattr(paddle.device, "is_compiled_with_cuda"):
            is_cuda = bool(paddle.device.is_compiled_with_cuda())
        elif hasattr(paddle, "is_compiled_with_cuda"):
            is_cuda = bool(paddle.is_compiled_with_cuda())
        if is_cuda:
            try:
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
        # empty_cache / teardown can leave default Place(undefined:0); re-pin for next init/inference.
        ensure_paddle_device_from_env()
    except Exception:
        pass


def get_gpu_memory_info() -> Tuple[float, float]:
    """Get GPU memory usage information.
    
    Returns
    -------
    Tuple[float, float]
        A tuple of (used_gb, total_gb). Returns (0.0, 0.0) if unable to query.
    
    Note
    ----
    Uses NVML (NVIDIA Management Library) for accurate memory readings.
    Falls back gracefully if NVML is not available.
    """
    try:
        # Try NVML first for accurate readings
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) == 2:
                used_mb = float(parts[0])
                total_mb = float(parts[1])
                return used_mb / 1024.0, total_mb / 1024.0
    except Exception:
        pass
    
    # Fallback: try pynvml if available
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / 1e9, info.total / 1e9
    except Exception:
        pass
    
    return 0.0, 0.0


def check_gpu_memory_and_downgrade(threshold_gb: float = 1.5) -> bool:
    """If GPU free memory is low, reduce OCR batch size and try to free cache.

    KnowMat keeps PP-StructureV3 / chemical re-OCR enabled by default; this helper
    does not set ``KNOWMAT_SKIP_PPSTRUCTURE_REFINE`` (legacy behavior removed).

    Parameters
    ----------
    threshold_gb : float
        Minimum free GPU memory in GB before triggering mitigations.
        Default is 1.5 GB, suitable for 8GB cards.

    Returns
    -------
    bool
        True if mitigations were applied, False otherwise.
    """
    used_gb, total_gb = get_gpu_memory_info()

    if total_gb <= 0:
        return False

    free_gb = total_gb - used_gb

    if free_gb < threshold_gb:
        logger.warning(
            "Low GPU memory detected: %.1f GB free of %.1f GB (threshold: %.1f GB). "
            "Reducing OCR_BATCH_SIZE and releasing Paddle cache (PP-StructureV3 remains enabled).",
            free_gb,
            total_gb,
            threshold_gb,
        )

        current_batch = os.getenv("OCR_BATCH_SIZE", "2")
        if current_batch != "1":
            os.environ["OCR_BATCH_SIZE"] = "1"
            logger.info("Reduced OCR_BATCH_SIZE to 1 due to low GPU memory")

        try_release_paddle_gpu_memory()

        return True

    return False


def log_gpu_memory_status(context: str = "") -> None:
    """Log current GPU memory status for debugging.
    
    Parameters
    ----------
    context : str
        Optional context string to include in the log message.
    """
    used_gb, total_gb = get_gpu_memory_info()
    
    if total_gb > 0:
        free_gb = total_gb - used_gb
        usage_pct = (used_gb / total_gb) * 100
        context_str = f" ({context})" if context else ""
        logger.info(
            "GPU Memory%s: %.1f/%.1f GB used (%.1f%%), %.1f GB free",
            context_str, used_gb, total_gb, usage_pct, free_gb
        )
    else:
        logger.debug("Unable to query GPU memory status")


def _ocr_infer_timeout_sec() -> float:
    """Seconds for optional OCR call timeout (0 = disabled). See module doc / README for caveats."""
    try:
        v = float(os.getenv("OCR_INFER_TIMEOUT_SEC", "0").strip())
    except ValueError:
        return 0.0
    return max(0.0, v)


def _invoke_ocr_inference(engine: Any, image_path: Path) -> Any:
    img = str(image_path)
    if hasattr(engine, "predict"):
        try:
            return engine.predict(img)
        except TypeError:
            return engine.predict(input=img)
        except (RuntimeError, ValueError, OSError):
            pass
    if hasattr(engine, "ocr"):
        try:
            return engine.ocr(img, cls=True)
        except TypeError:
            return engine.ocr(img)
    if callable(engine):
        return engine(img)
    raise RuntimeError("OCR engine does not expose a known inference method.")


def _run_ocr(engine: Any, image_path: Path) -> Any:
    timeout = _ocr_infer_timeout_sec()
    if timeout <= 0:
        return _invoke_ocr_inference(engine, image_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_invoke_ocr_inference, engine, image_path)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error(
                "OCR inference exceeded OCR_INFER_TIMEOUT_SEC=%.1fs for %s; returning None. "
                "The worker thread may keep running inside native/CUDA code and cannot be "
                "terminated from Python.",
                timeout,
                image_path,
            )
            try_release_paddle_gpu_memory()
            return None


def supports_batch_predict(engine: Any) -> bool:
    """Check if the engine supports batch prediction.
    
    For PaddleOCR-VL 1.5, also check for predict_batch method existence
    or try to infer from class name and available methods.
    """
    # Direct check for explicit batch method
    if hasattr(engine, "predict_batch"):
        return True
    
    # PaddleOCR-VL specific check: if it has restructure_pages, it's the VL version
    # which should support batch via predict or predict_batch
    if hasattr(engine, "restructure_pages"):
        return True
    
    # Fallback: check predict signature for list annotation
    if hasattr(engine, "predict"):
        try:
            import inspect

            sig = inspect.signature(engine.predict)
            for param in sig.parameters.values():
                if param.annotation in ("list", "List", List):
                    return True
                # Also check for list[str] or List[str] patterns
                param_str = str(param.annotation)
                if "list" in param_str.lower() and "str" in param_str.lower():
                    return True
        except (ValueError, TypeError):
            pass
    
    return False


def run_ocr_batch(engine: Any, image_paths: List[Path], batch_size: int) -> List[Any]:
    results: List[Any] = [None] * len(image_paths)
    timeout = _ocr_infer_timeout_sec()
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        str_paths = [str(p) for p in chunk]
        try:
            if timeout > 0:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(engine.predict_batch, str_paths)
                    try:
                        batch_raw = fut.result(timeout=timeout)
                    except FuturesTimeoutError:
                        logger.error(
                            "predict_batch exceeded OCR_INFER_TIMEOUT_SEC=%.1fs; "
                            "falling back to per-image OCR for this chunk.",
                            timeout,
                        )
                        try_release_paddle_gpu_memory()
                        batch_raw = [_run_ocr(engine, p) for p in chunk]
            else:
                batch_raw = engine.predict_batch(str_paths)
        except (TypeError, AttributeError):
            batch_raw = [_run_ocr(engine, p) for p in chunk]
        for j, raw in enumerate(batch_raw):
            results[start + j] = raw
    return results


def run_ocr_sequential(engine: Any, image_paths: List[Path]) -> List[Any]:
    """Run OCR on each image one at a time.

    A shared Paddle/PaddleOCR engine instance is not safe for concurrent
    inference from multiple threads; use batch inference via :func:`run_ocr_batch`
    when the engine supports it.
    """
    return [_run_ocr(engine, p) for p in image_paths]


def collect_text(obj: Any, out: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            out.append(text)
        return
    if isinstance(obj, dict):
        for key in ("text", "rec_text", "ocr_text", "content", "transcription"):
            val = obj.get(key)
            if isinstance(val, str):
                val = val.strip()
                if val:
                    out.append(val)
        for val in obj.values():
            collect_text(val, out)
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and isinstance(obj[1], (list, tuple)) and len(obj[1]) > 0 and isinstance(obj[1][0], str):
            text = obj[1][0].strip()
            if text:
                out.append(text)
            return
        for item in obj:
            collect_text(item, out)


def _as_box_points(obj: Any) -> Optional[List[List[float]]]:
    if obj is None:
        return None
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
    except ImportError:
        pass
    if not isinstance(obj, (list, tuple)) or len(obj) < 4:
        return None
    first = obj[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        pts: List[List[float]] = []
        for p in obj[:4]:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                return None
            pts.append([float(p[0]), float(p[1])])
        return pts
    if len(obj) >= 4 and all(isinstance(x, (int, float)) for x in obj[:4]):
        x1, y1, x2, y2 = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return None


def _box_cx_cy_h(pts: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    h = max(ys) - min(ys) + 1e-6
    return (sum(xs) / len(xs), sum(ys) / len(ys), h)


def _paddle_rec_text(rec: Any) -> str:
    if isinstance(rec, (list, tuple)) and rec and isinstance(rec[0], str):
        return rec[0].strip()
    return ""


def _paddle_rec_is_valid(rec: Any) -> bool:
    return isinstance(rec, (list, tuple)) and len(rec) >= 1 and isinstance(rec[0], str)


def _collect_paddleocr_line_entries(obj: Any, out: List[Tuple[float, float, float, str]]) -> None:
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        box, rec = obj[0], obj[1]
        pts = _as_box_points(box)
        if pts and _paddle_rec_is_valid(rec):
            text = _paddle_rec_text(rec)
            if text:
                cx, cy, h = _box_cx_cy_h(pts)
                out.append((cx, cy, h, text))
            return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_paddleocr_line_entries(item, out)


def paddleocr_raw_to_lines(raw: Any) -> List[str]:
    """Build reading-ordered lines from classic PaddleOCR ``ocr()`` output (bbox + rec tuple).

    Falls back to :func:`collect_text` order when no box/text pairs are found.
    """
    entries: List[Tuple[float, float, float, str]] = []
    _collect_paddleocr_line_entries(raw, entries)
    if not entries:
        fallback: List[str] = []
        collect_text(raw, fallback)
        return fallback
    if len(entries) == 1:
        return [entries[0][3]]
    heights = [e[2] for e in entries]
    med_h = float(median(heights))
    band = max(8.0, med_h * 0.45)
    entries.sort(key=lambda e: (e[1], e[0]))
    line_groups: List[List[Tuple[float, float, float, str]]] = []
    for e in entries:
        cy = e[1]
        if not line_groups:
            line_groups.append([e])
            continue
        cur = line_groups[-1]
        ref_cy = sum(x[1] for x in cur) / len(cur)
        if abs(cy - ref_cy) <= band:
            cur.append(e)
        else:
            line_groups.append([e])
    out_lines: List[str] = []
    for grp in line_groups:
        grp.sort(key=lambda x: x[0])
        out_lines.append(" ".join(x[3] for x in grp))
    return out_lines


def normalize_lines(lines: List[str]) -> List[str]:
    normalized: List[str] = []
    prev = ""
    for line in lines:
        text = re.sub(r"\s+", " ", line).strip()
        if not text or text == prev:
            continue
        normalized.append(text)
        prev = text
    return normalized
