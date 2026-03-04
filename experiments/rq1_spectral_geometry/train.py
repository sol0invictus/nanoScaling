"""
RQ1: Spectral Geometry — Muon vs. AdamW weight matrix structure.

All training logic lives in the root train.py.
Enable spectral logging via YAML config: enable_spectral_logging: true
"""
import subprocess, sys, os
root = os.path.join(os.path.dirname(__file__), '..', '..', 'train.py')
sys.exit(subprocess.call([sys.executable, os.path.abspath(root)] + sys.argv[1:]))
