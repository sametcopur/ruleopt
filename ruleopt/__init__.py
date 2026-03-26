from importlib import import_module

__all__ = [
    "RUGClassifier",
    "RUXClassifier",
    "RUXLGBMClassifier",
    "RUXXGBClassifier",
    "Explainer",
]


def __getattr__(name):
    if name in {
        "RUGClassifier",
        "RUXClassifier",
        "RUXLGBMClassifier",
        "RUXXGBClassifier",
    }:
        return getattr(import_module(".estimator", __name__), name)
    if name == "Explainer":
        return getattr(import_module(".explainer", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
