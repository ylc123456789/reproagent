"""Compatibility shim — hardware context moved to runtime.hardware."""

from .runtime.hardware import collect_hardware_text

__all__ = ["collect_hardware_text"]
