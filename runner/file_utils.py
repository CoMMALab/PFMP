import itertools, json, os, tempfile, time, traceback
from pathlib import Path
from typing import Any, Sequence, Tuple, NamedTuple, Type
import numpy as np

def is_namedtuple_instance(obj) -> bool:
    return (
        isinstance(obj, tuple) and
        hasattr(obj, '_asdict') and
        hasattr(obj, '_fields')
    )

def _to_jsonable(x):
    """Recursively convert a pytree to JSON-safe types."""
    import numpy as _np
    import jax.numpy as _jnp
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, _np.generic):  # numpy scalar -> python scalar
        return x.item()
    if isinstance(x, (_np.ndarray, _jnp.ndarray)):
        return x.tolist()
    # namedtuple must be above tuple, since namedtuple is also a tuple
    if is_namedtuple_instance(x):
        return _to_jsonable(x._asdict())
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    return str(x)  # fallback

def _atomic_write_json(obj, path: Path, indent: int = 2) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False, allow_nan=True)
            f.write("\n")
        os.replace(tmp, str(path))
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def save_trajs_npz_named(trajs: Any, path) -> None:
    """
    If `trajs` is a NamedTuple, save fields by name.
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for f in trajs._fields:
        arrays[f] = np.asarray(getattr(trajs, f))
    np.savez_compressed(path, **arrays)

def save_traj_costs_npz_named(traj_costs: Any, path) -> None:
    """
    If `trajs` is a NamedTuple, save fields by name.
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    arrays['traj_costs'] = np.asarray(traj_costs)
    np.savez_compressed(path, **arrays)


def load_trajs_npz_named(path, as_namedtuple: Type | None = None):
    """
    Load .npz to:
      - dict[str, np.ndarray] (default), or
      - a NamedTuple instance if `as_namedtuple` is given (fields must match keys).
    """
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}

    if as_namedtuple is not None:
        # Ensure fields exist (no silent mismatch)
        missing = [f for f in as_namedtuple._fields if f not in d]
        if missing:
            raise KeyError(f"Missing fields in NPZ for {as_namedtuple.__name__}: {missing}")
        # Extra keys are ignored
        return as_namedtuple(**{f: d[f] for f in as_namedtuple._fields})

    return d  # plain dict
