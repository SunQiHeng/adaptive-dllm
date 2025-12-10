try:
    from .core.modeling_dream import DreamModel
    from .core.tokenization_dream import DreamTokenizer
    __all__ = ["DreamModel", "DreamTokenizer"]
except ImportError:
    # Allow individual submodules to be imported even if main classes fail
    pass