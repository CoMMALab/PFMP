"""Stub for the two names `dial_core` imports but never uses in the code paths
we test (they serve its CLI entry point). Not upstream code."""


def get_example_path(*args, **kwargs):        # pragma: no cover
    raise NotImplementedError("vendored for import only")


def load_dataclass_from_dict(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("vendored for import only")
