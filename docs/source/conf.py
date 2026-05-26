import os
import sys

sys.path.insert(0, os.path.abspath("../../arclib-numerics-py"))

project = "arclib"
copyright = ""
author = ""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

numericss_path = ["_numericss"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
