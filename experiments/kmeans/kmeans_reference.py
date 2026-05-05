"""Readable k-means reference implementation for small validation cases."""

from __future__ import annotations

from .kmeans_numpy import kmeans_numpy_naive


def kmeans_reference(*args, **kwargs):
    """Use the textbook broadcast implementation as the local oracle."""
    return kmeans_numpy_naive(*args, **kwargs)
