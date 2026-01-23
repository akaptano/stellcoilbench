import os
import sys
import subprocess
from pathlib import Path

project = "StellCoilBench"
author = "Alan Kaptanoglu"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

# Enable raw HTML in RST tables
html_use_smartypants = False

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

# Run update-db before building documentation to ensure leaderboard is up to date
def run_update_db(app, config):
    """Run update-db command before building documentation."""
    # Get the repo root (parent of docs directory)
    repo_root = Path(__file__).parent.parent
    
    # Try to run update-db command
    try:
        # Change to repo root to run the command
        result = subprocess.run(
            ["stellcoilbench", "update-db"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,  # Don't fail build if update-db fails
        )
        if result.returncode == 0:
            print("Successfully updated leaderboard database before building docs.")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"Warning: update-db returned non-zero exit code: {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
    except FileNotFoundError:
        print("Warning: 'stellcoilbench' command not found. Skipping update-db.")
        print("Make sure stellcoilbench is installed: pip install -e .")
    except Exception as e:
        print(f"Warning: Failed to run update-db: {e}")
        print("Continuing with documentation build anyway...")

def setup(app):
    """Register event handlers for Sphinx."""
    # Run update-db when builder is initialized (before reading source files)
    app.connect("config-inited", run_update_db)
