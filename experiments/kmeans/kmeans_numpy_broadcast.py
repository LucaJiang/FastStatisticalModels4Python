"""Compatibility wrapper for the broadcast k-means implementation."""

from __future__ import annotations

from .kmeans_numpy import kmeans_numpy_naive


def kmeans_numpy_broadcast(*args, **kwargs):
    return kmeans_numpy_naive(*args, **kwargs)
