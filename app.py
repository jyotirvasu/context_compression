"""
Web server for Context Compression Showcase.
Serves the HTML comparison page and provides an API to re-run the pipeline.
"""

import os
import subprocess
import sys
from flask import Flask, send_file, jsonify

app = Flask(__name__)

SHOWCASE_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "showcase_results.html")


def ensure_showcase_exists():
    """Generate showcase HTML if it doesn't exist."""
    if not os.path.exists(SHOWCASE_HTML):
        subprocess.run(
            [sys.executable, "showcase.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
        )


@app.route("/")
def index():
    """Serve the showcase comparison HTML page."""
    ensure_showcase_exists()
    return send_file(SHOWCASE_HTML)


@app.route("/health")
def health():
    """Health check endpoint for Kubernetes probes."""
    return jsonify({"status": "healthy"}), 200


@app.route("/run")
def run_pipeline():
    """Re-run the showcase pipeline and return the updated HTML."""
    try:
        subprocess.run(
            [sys.executable, "showcase.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=True,
            text=True,
        )
        return send_file(SHOWCASE_HTML)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Pipeline failed", "details": e.stderr}), 500


if __name__ == "__main__":
    ensure_showcase_exists()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
