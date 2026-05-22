"""Forward hook manager for WiFlow intermediate feature extraction.

Provides a context-manager-based hook system that attaches forward hooks to
named submodules of a WiFlowModel, collects their outputs, and automatically
removes all hooks on context exit — zero side effects on the model after use.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import torch
from torch import nn


class WiFlowHookContext:
    """Registers forward hooks on WiFlow submodules and collects outputs.

    Usage::

        ctx = WiFlowHookContext(model)
        ctx.register("spatial_encoder.antenna_mixer")
        with torch.no_grad():
            model(x)
        mixer_out = ctx.storage["spatial_encoder.antenna_mixer"]
        ctx.remove_hooks()
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._storage: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_hook(self, name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            self._storage[name] = output
        return hook

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, module_path: str) -> None:
        """Register a hook on a dot-separated submodule path.

        Example: ``ctx.register("spatial_encoder.antenna_mixer")``
        """
        module = self._model
        for part in module_path.split("."):
            module = getattr(module, part)
        handle = module.register_forward_hook(self._make_hook(module_path))
        self._handles.append(handle)

    def register_many(self, paths: list[str]) -> None:
        """Convenience to register multiple paths at once."""
        for path in paths:
            self.register(path)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def storage(self) -> dict[str, Any]:
        return self._storage

    def get(self, name: str, default: Any = None) -> Any:
        return self._storage.get(name, default)

    def get_tensor(self, name: str) -> torch.Tensor | None:
        """Return a plain tensor if the stored value is one; otherwise None."""
        val = self._storage.get(name)
        if isinstance(val, torch.Tensor):
            return val
        return None

    def get_attention_weights(self, name: str) -> torch.Tensor | None:
        """Extract attention weights from a MultiheadAttention hook.

        MultiheadAttention returns ``(attn_output, attn_weights)`` as a tuple.
        This helper extracts the weights (second element).
        """
        val = self._storage.get(name)
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            weights = val[1]
            if isinstance(weights, torch.Tensor):
                return weights
        return None

    def clear(self) -> None:
        self._storage.clear()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------


@contextmanager
def wiflow_hooks(
    model: nn.Module,
    hook_points: list[str] | None = None,
) -> WiFlowHookContext:
    """Context manager that registers hooks on entry, removes them on exit.

    Args:
        model: A WiFlowModel (or any nn.Module with named children).
        hook_points: Dot-separated submodule paths to hook.  If ``None``,
            no hooks are pre-registered; the caller can use ``ctx.register()``.

    Yields:
        ``WiFlowHookContext`` with collected outputs.
    """
    ctx = WiFlowHookContext(model)
    if hook_points:
        ctx.register_many(hook_points)
    try:
        yield ctx
    finally:
        ctx.remove_hooks()