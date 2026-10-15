"""usage: same arguments as run.py; GRAD_CKPTS / GRAD_NB / GRAD_OUT env (see exp_grad.py). Runs the A1 per-origin gradient diagnostic."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
np.Inf = np.inf
from patchmem.exp_grad import Exp_Grad
src = (ROOT / 'run.py').read_text()
assert src.count('Exp = Exp_Long_Term_Forecast') == 1
sys.argv[0] = str(ROOT / 'run.py')
exec(compile(src.replace('Exp = Exp_Long_Term_Forecast', 'Exp = Exp_Grad'), str(ROOT / 'run.py'), 'exec'), {'__name__': '__main__', '__file__': str(ROOT / 'run.py'), 'Exp_Grad': Exp_Grad})
