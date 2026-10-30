"""usage: same arguments as run.py; env TEFL_* (see exp_a0tefl.py). Runs the official run.py with Exp = Exp_A0Tefl."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
np.Inf = np.inf
from patchmem.exp_a0tefl import Exp_A0Tefl as Exp_A1
src = (ROOT / 'run.py').read_text()
assert src.count('Exp = Exp_Long_Term_Forecast') == 1
sys.argv[0] = str(ROOT / 'run.py')
exec(compile(src.replace('Exp = Exp_Long_Term_Forecast', 'Exp = Exp_A1'), str(ROOT / 'run.py'), 'exec'), {'__name__': '__main__', '__file__': str(ROOT / 'run.py'), 'Exp_A1': Exp_A1})
