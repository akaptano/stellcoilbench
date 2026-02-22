import os
import sys

project = "StellCoilBench"
author = "Alan Kaptanoglu"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

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

def setup(app):
    """Register event handlers for Sphinx."""
    # Add custom CSS for better table display (replaces deprecated html_css_files)
    app.add_css_file("custom.css")
    # Add leaderboard sort script for clickable column headers
    app.add_js_file("leaderboard_sort.js")
