import os
import sys
project = "StellCoilBench"
author = "Alan Kaptanoglu"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

# Add custom CSS for better table display
html_css_files = ["custom.css"]

# Configure ReadTheDocs theme for wider content - allow full width expansion
html_theme_options = {
    "body_max_width": "none",  # Remove max-width constraint
}

# Allow content to expand beyond viewport width if needed
html_static_path = ["_static"]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

sys.path.insert(0, os.path.abspath(".."))
