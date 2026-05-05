"""Compatibility wrapper for the matmul k-means implementation."""

from __future__ import annotations

from .kmeans_numpy import kmeans_numpy_smart


def kmeans_numpy_matmul(*args, **kwargs):
    return kmeans_numpy_smart(*args, **kwargs)
