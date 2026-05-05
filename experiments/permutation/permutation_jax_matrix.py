"""Optional JAX matrix path for permutation-test quick checks."""

from __future__ import annotations

import numpy as np

from .permutation_numpy import contrast_matrix
from .permutation_reference import observed_statistics, p_values_from_null


def jax_available() -> tuple[bool, str]:
    try:
        import jax  # type: ignore

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local env
        return False, repr(exc)
    try:
        return True, f"backend={jax.default_backend()}, devices={jax.devices()}"
    except Exception as exc:  # pragma: no cover
        return True, f"available, metadata_error={exc!r}"


def jax_matrix_p_values(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    r: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    jax.config.update("jax_enable_x64", True)
    observed = observed_statistics(x, labels)
    w = contrast_matrix(labels, r=r, seed=seed)
    null_device = jnp.asarray(w) @ jnp.asarray(x)
    null = np.asarray(null_device.block_until_ready())
    return observed, null, p_values_from_null(observed, null)
