import os
import sys
from unittest.mock import MagicMock
from sphinx.application import Sphinx


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
            return MagicMock()



project_path = os.path.abspath("../..")
sys.path.insert(0, project_path)


project = "ruleopt"
copyright = "2024, Samet Çopur"
author = "Samet Çopur"
version = '1.0.1'
release = '1.0.1'
autodoc_member_order = "bysource"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "myst_parser",
]

autodoc_typehints = "none"

autodoc_mock_imports = ["ruleopt.aux_classes.aux_classes"]

python_use_unqualified_type_names = True

autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

def setup(app: Sphinx):
    print("setup")
    def remove_all_parameters(app, what, name, obj, options, signature, return_annotation):
        # Yalnızca sınıflar için imzayı değiştir
        if what == "class":
            # Tüm parametreleri kaldırmak için imzayı boş bir string olarak ayarla
            return ("", return_annotation)
        # Diğer tüm durumlar için orijinal imzayı kullan
        return (signature, return_annotation)

    app.connect("autodoc-process-signature", remove_all_parameters)
