import argparse, json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.runtime import load_model, StaticGraph
import cv2, numpy as np, torch
from PIL import Image, ImageDraw
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='weights/kad_cnet_s_dior_mgl.pt')
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--data', help='Optional dataset YAML; classes are read from the checkpoint')
    p.add_argument('--backend', choices=['raw', 'optimized', 'graph'], default='raw')
    p.add_argument('--config')
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--conf-thres', type=float, default=0.25)
    p.add_argument('--iou-thres', type=float, default=0.4)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, default=Path('outputs/detect'))
    p.add_argument('--checkpoint-key', choices=['auto', 'ema', 'model'], default='auto')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    model = load_model(a.weights, a.backend, a.device, a.config, checkpoint_key=a.checkpoint_key)
    runner = None
    records = []
    files = sorted(a.source.glob('*')) if a.source.is_dir() else [a.source]
    files = [f for f in files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}]
    if not files:
        raise FileNotFoundError(a.source)
    for f in files:
        original = cv2.imread(str(f))
        if original is None:
            raise ValueError(f'Cannot read image: {f}')
        processed = letterbox(original, new_shape=(a.imgsz, a.imgsz), auto=False, stride=32)[0]
        x = (
            torch.from_numpy(np.ascontiguousarray(processed[:, :, ::-1].transpose(2, 0, 1)))
            .to(a.device)
            .float()
            .unsqueeze(0)
            / 255.0
        )
        with torch.no_grad():
            if a.backend == 'graph':
                if runner is None:
                    runner = StaticGraph(model, x)
                prediction = runner(x)
            else:
                prediction = model(x)
        value = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        detections = non_max_suppression(value, a.conf_thres, a.iou_thres, multi_label=False, max_det=a.max_det)[0]
        detections[:, :4] = scale_boxes(x.shape[2:], detections[:, :4], original.shape).round()
        array = detections.detach().cpu().tolist()
        image = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        lines = []
        for xmin, ymin, xmax, ymax, confidence, cls in array:
            draw.rectangle((xmin, ymin, xmax, ymax), outline='#e63946', width=2)
            draw.text((xmin, max(0, ymin - 12)), f'{model.names[int(cls)]} {confidence:.3f}', fill='#e63946')
            box = xyxy2xywh(torch.tensor([[xmin, ymin, xmax, ymax]])).flatten() / torch.tensor(
                [original.shape[1], original.shape[0], original.shape[1], original.shape[0]]
            )
            lines.append(' '.join([str(int(cls)), *[f'{v:.9g}' for v in box.tolist()], f'{confidence:.9g}']))
        image.save(a.output / (f.stem + '.png'))
        (a.output / (f.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        records.append(
            {
                'image_id': f.stem,
                'backend': a.backend,
                'count': len(array),
                'detections_xyxy_conf_class': array,
                'tensor_shape': list(x.shape),
                'conf': a.conf_thres,
                'nms_iou': a.iou_thres,
                'multi_label': False,
            }
        )
    (a.output / 'detections.json').write_text(json.dumps(records, indent=2))
    print(
        json.dumps(
            {
                'backend': a.backend,
                'images': len(records),
                'detections': sum(r['count'] for r in records),
                'output': str(a.output),
            }
        )
    )


if __name__ == '__main__':
    main()
