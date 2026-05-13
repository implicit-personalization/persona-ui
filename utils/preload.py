from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Iterable

logger = logging.getLogger(__name__)

_started: set[tuple[str, ...]] = set()
_lock = threading.Lock()


def _warm_imports(
    modules: tuple[str, ...],
    functions: tuple[str, ...],
    delay_seconds: float,
) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception:
            logger.debug("Background preload failed for %s", module, exc_info=True)
    for function_path in functions:
        try:
            module_name, function_name = function_path.split(":", 1)
            function = getattr(importlib.import_module(module_name), function_name)
            function()
        except Exception:
            logger.debug(
                "Background preload failed for %s", function_path, exc_info=True
            )


def preload_once(
    name: str,
    *,
    modules: Iterable[str] = (),
    functions: Iterable[str] = (),
    delay_seconds: float = 0.25,
) -> None:
    """Warm small predictable costs on a daemon thread after the visible render.

    Keep this limited to imports and tiny local metadata. Avoid model
    construction, Hub requests, and Streamlit cache population because those can
    steal enough CPU or I/O to make the visible page feel slower.
    """

    module_tuple = tuple(dict.fromkeys(modules))
    function_tuple = tuple(dict.fromkeys(functions))
    if not module_tuple and not function_tuple:
        return

    key = (name, *module_tuple, *function_tuple)
    with _lock:
        if key in _started:
            return
        _started.add(key)

    thread = threading.Thread(
        target=_warm_imports,
        args=(module_tuple, function_tuple, delay_seconds),
        name=f"persona-ui-preload-{name}",
        daemon=True,
    )
    thread.start()
