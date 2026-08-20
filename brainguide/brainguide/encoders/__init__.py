"""Encoder registry — build any BrainEncoder by name from a config dict.

``brainguide/__init__.py`` promises "swap encoders by name in a YAML config;
nothing else in the loop changes." This module is where that name -> class
mapping lives, so the runner never imports a concrete encoder directly.

    build_encoder("braindive_clip", num_voxels=1000)   # runnable today
    build_encoder("mock")                              # random-head smoke test
    build_encoder("universal", checkpoint="...")       # Paper 1 (needs vendored code)
"""
from .base import BrainEncoder, make_brain_tweak

__all__ = ["BrainEncoder", "make_brain_tweak", "build_encoder", "ENCODER_REGISTRY"]


def _build_braindive_clip(**kw):
    from .braindive_clip import BrainDiVECLIPEncoder
    return BrainDiVECLIPEncoder(**kw)


def _build_mock(**kw):
    # "mock" is braindive_clip with a random readout head (weights_path=None).
    from .braindive_clip import BrainDiVECLIPEncoder
    kw.pop("weights_path", None)
    return BrainDiVECLIPEncoder(weights_path=None, **kw)


def _build_universal(**kw):
    from .universal import UniversalEncoder
    return UniversalEncoder(**kw)


#: name -> zero-arg-safe builder. Builders import lazily so that, e.g., asking
#: for "braindive_clip" never triggers importing the universal adapter.
ENCODER_REGISTRY = {
    "braindive_clip": _build_braindive_clip,
    "mock": _build_mock,
    "universal": _build_universal,
}


def build_encoder(name: str, **kwargs) -> BrainEncoder:
    """Instantiate the encoder registered under ``name`` with ``**kwargs``."""
    try:
        builder = ENCODER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown encoder {name!r}; choose from {sorted(ENCODER_REGISTRY)}"
        )
    return builder(**kwargs)
