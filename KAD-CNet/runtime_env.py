"""Keep build and cache files inside the repo (outputs/.runtime) and disable YOLO auto-install."""

import os, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def configure():
    runtime = Path(os.environ.get('KADC_RUNTIME_DIR', ROOT / 'outputs/.runtime'))
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / 'tmp').mkdir(exist_ok=True)
    temporary = os.environ.get('TMPDIR', '')
    if not temporary or len(temporary) > 60:
        temporary = tempfile.mkdtemp(prefix='kadc-', dir='/tmp')
    os.environ['TMPDIR'] = temporary
    tempfile.tempdir = None
    for key, suffix in [
        ('MPLCONFIGDIR', 'matplotlib'),
        ('YOLOV5_CONFIG_DIR', 'yolo'),
        ('TORCH_HOME', 'torch'),
        ('TORCH_EXTENSIONS_DIR', 'extensions'),
        ('TRITON_CACHE_DIR', 'triton'),
        ('CUDA_CACHE_PATH', 'cuda'),
        ('XDG_CACHE_HOME', 'xdg'),
    ]:
        os.environ.setdefault(key, str(runtime / suffix))
    os.environ['YOLOv5_AUTOINSTALL'] = 'False'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('KADC_DCN16_BUILD_DIR', str(runtime / 'dcn16_s'))
    os.environ.setdefault('KADC_GEMM_SILU_BUILD', str(runtime / 'gemm_s'))
    os.environ.setdefault('MAX_JOBS', '2')
    return runtime
