"""Compare the accelerated KAD-Conv with the plain PyTorch version on random layers and time both.

    python check_fast.py                 # eager timing
    python check_fast.py --graph         # timing with CUDA Graph, as in the paper

Use --allow-untested-gpu on GPUs other than A100.
"""
import argparse

import torch

from kadconv import KADConv, disable_fast_inference, enable_fast_inference


def runner(layer, x, graph):
    if not graph:
        return lambda: layer(x)
    static = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            layer(static)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        layer(static)
    return g.replay


def timed(run, iters):
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--allow-untested-gpu', action='store_true')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    # (channels, feature size): the first three give exactly the DSC shapes of KAD-CNet-S at 512 input
    # (handled by the CUDA kernel); the others fall back to the Triton kernel
    shapes = [(32, 64), (48, 32), (64, 16), (64, 8), (96, 40), (24, 100)]
    print(f'{"C":>4} {"HxW":>8} {"max abs err":>12} {"max rel err":>12} {"plain ms":>9} {"fast ms":>8}')
    all_ok = True
    with torch.no_grad():
        for c, s in shapes:
            layer = KADConv(c).cuda().eval()
            x = torch.randn(1, c, s, s, device='cuda')
            ref = layer(x).clone()
            t_plain = timed(runner(layer, x, args.graph), args.iters)
            enable_fast_inference(layer, allow_untested_gpu=args.allow_untested_gpu)
            out = layer(x).clone()
            t_fast = timed(runner(layer, x, args.graph), args.iters)
            disable_fast_inference(layer)
            abs_err = (out - ref).abs().max().item()
            rel_err = abs_err / max(ref.abs().max().item(), 1e-12)
            ok = abs_err <= 1e-3 and rel_err <= 2e-6
            all_ok &= ok
            print(f'{c:>4} {f"{s}x{s}":>8} {abs_err:>12.2e} {rel_err:>12.2e} {t_plain:>9.3f} {t_fast:>8.3f}  '
                  f'{"ok" if ok else "FAIL"}')
    raise SystemExit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
