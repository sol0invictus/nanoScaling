"""
RQ2: Normalization ablations — can Muon train without LayerNorm?

All training logic lives in the root train.py.
Enable stability monitoring via YAML config: enable_stability_monitor: true
"""
import subprocess, sys, os
root = os.path.join(os.path.dirname(__file__), '..', '..', 'train.py')
sys.exit(subprocess.call([sys.executable, os.path.abspath(root)] + sys.argv[1:]))
