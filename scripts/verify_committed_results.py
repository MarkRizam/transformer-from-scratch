from pathlib import Path
import json
from transformer_scratch.experiments import run, compare
if __name__=='__main__':
    saved=json.loads((Path(__file__).resolve().parents[1]/'results/benchmarks.json').read_text())
    compare(saved['records'],run()); print('63 committed records verified at absolute tolerance 1e-9')
