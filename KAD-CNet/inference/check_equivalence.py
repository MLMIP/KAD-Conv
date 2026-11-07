import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.runtime import load_model, StaticGraph, leaves, clone_output
import torch


def compare(expected, actual):
    a, b = leaves(expected), leaves(actual)
    assert len(a) == len(b)
    absmax = relmax = 0.0
    for x, y in zip(a, b):
        assert x.shape == y.shape
        error = float((x - y).abs().max())
        absmax = max(absmax, error)
        relmax = max(relmax, error / max(float(x.abs().max()), 1e-12))
    return {
        'leaves': len(a),
        'global_max_abs': absmax,
        'global_max_rel': relmax,
        'pass': absmax <= 1e-3 and relmax <= 2e-6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/kad_cnet_s_dior_mgl.pt')
    ap.add_argument('--config')
    ap.add_argument('--imgsz', type=int, default=512)
    ap.add_argument('--output', type=Path, default=Path('outputs/equivalence.json'))
    ap.add_argument('--checkpoint-key', choices=['auto', 'ema', 'model'], default='auto')
    a = ap.parse_args()
    torch.manual_seed(20260803)
    raw = load_model(a.weights, 'raw', checkpoint_key=a.checkpoint_key)
    optimized = load_model(a.weights, 'optimized', config=a.config, checkpoint_key=a.checkpoint_key)
    xs = [torch.randn(1, 3, a.imgsz, a.imgsz, device='cuda') for _ in range(3)]
    rows = []
    with torch.no_grad():
        for _ in range(5):
            raw(xs[0])
            optimized(xs[0])
        graph = StaticGraph(optimized, xs[0])
        for i, x in enumerate(xs):
            rawout = clone_output(raw(x))
            fastout = clone_output(optimized(x))
            graphout = graph(x)
            torch.cuda.synchronize()
            rows.append(
                {
                    'input': i,
                    'raw_vs_optimized': compare(rawout, fastout),
                    'optimized_vs_graph': compare(fastout, graphout),
                }
            )
    report = {
        'seed': 20260803,
        'dtype': 'float32',
        'tf32': False,
        'shape': [1, 3, a.imgsz, a.imgsz],
        'comparisons': rows,
        'pass': all(r[k]['pass'] for r in rows for k in ['raw_vs_optimized', 'optimized_vs_graph']),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert report['pass'], 'Numerical equivalence failed'


if __name__ == '__main__':
    main()
