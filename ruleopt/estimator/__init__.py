from importlib import import_module

__all__ = [
    "RUGClassifier",
    "RUXClassifier",
    "RUXLGBMClassifier",
    "RUXXGBClassifier",
]

_MODULE_MAP = {
    "RUGClassifier": ".rug",
    "RUXClassifier": ".rux",
    "RUXLGBMClassifier": ".lightgbm_",
    "RUXXGBClassifier": ".xgboost_",
}


def __getattr__(name):
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
