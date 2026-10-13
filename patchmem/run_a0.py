"""Official run.py unchanged, plus a numpy-2 shim (official utils/tools.py uses np.Inf, removed in numpy 2.0)."""
import sys, runpy
from pathlib import Path
import numpy as np
np.Inf = np.inf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
sys.argv[0] = str(ROOT / 'run.py')
runpy.run_path(str(ROOT / 'run.py'), run_name='__main__')
