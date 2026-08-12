"""Compatibility shim — dataset cache bridging moved to runtime.dataset_cache."""

from .runtime.dataset_cache import prepare_dataset_links, render_dataset_block, scan_dataset_roots

__all__ = ["prepare_dataset_links", "render_dataset_block", "scan_dataset_roots"]
