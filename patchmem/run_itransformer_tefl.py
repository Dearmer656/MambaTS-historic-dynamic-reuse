"""Run iTransformer with the original TEFL correction and training schedule."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
np.Inf = np.inf
from patchmem.exp_itransformer_tefl import Exp_iTransformerTefl as Exp_TEFL

source = (ROOT / 'run.py').read_text()
assert source.count('Exp = Exp_Long_Term_Forecast') == 1
sys.argv[0] = str(ROOT / 'run.py')
exec(compile(source.replace('Exp = Exp_Long_Term_Forecast', 'Exp = Exp_TEFL'),
             str(ROOT / 'run.py'), 'exec'),
     {'__name__': '__main__', '__file__': str(ROOT / 'run.py'), 'Exp_TEFL': Exp_TEFL})
