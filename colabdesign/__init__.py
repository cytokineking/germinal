import os as _os

# Ensure subpackages (af, mpnn, shared, etc.) under ./colabdesign are discoverable
__path__ = [*_os.path.join(_os.path.dirname(__file__), 'colabdesign'), *__path__]  # type: ignore[name-defined]

# Re-export expected top-level symbols for backward compatibility
from .colabdesign import mk_afdesign_model  # noqa: F401
from .colabdesign.shared.utils import clear_mem  # noqa: F401


