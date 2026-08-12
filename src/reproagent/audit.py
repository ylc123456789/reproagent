"""Compatibility shim — environment audit moved to runtime.audit."""

from .runtime.audit import audit_environment

__all__ = ["audit_environment"]
