"""SAN open-vocab segmentation, patched for nuScenes-mini + subset testing.

Differences from main.py:
  * loads an explicit --version (default v1.0-mini) and iterates that version's scenes
    (the original ranged over numeric scene ids + hardcoded v1.0-trainval),
  * --scenes / --max_keyframes for quick subset runs, resumable (skips existing .png),
  * --fixed_vocab uses SAN's built-in nuScenes vocabulary (no LVLM step needed).
Output: <output_root>/samples/<CAM>/<image>.png  (per-pixel class index into the vocabulary).
"""
import os
import argparse

import cv2
from nuscenes.nuscenes import NuScenes
from predict import Predictor
from main import fixed_vocabulary, model_cfg, model_name, load_txt  # reuse defs (import is side-effect free)

CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']


def parse_config():
    p = argparse.ArgumentParser()
    p.add_argument('--fixed_vocab', action='store_true')
    p.add_argument('--vocab_root', default='data/occ3d/qwen_texts')
    p.add_argument('--data_root', default='data/occ3d')
    p.add_argument('--output_root', default='data/occ3d/san_qwen_scene')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--scenes', default='', help='comma-separated scene names; default = all in version')
    p.add_argument('--max_keyframes', type=int, default=0, help='0 = all; else cap key-frames per scene')
    return p.parse_args()


def main():
    args = parse_config()
    predictor = Predictor(**model_cfg[model_name])
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    scenes = list(nusc.scene)
    if args.scenes:
        want = {s.strip() for s in args.scenes.split(',') if s.strip()}
        scenes = [s for s in scenes if s['name'] in want]
    print(f"SAN over {len(scenes)} scenes (version {args.version}, fixed_vocab={args.fixed_vocab})", flush=True)

    if args.fixed_vocab:
        vocabulary = fixed_vocabulary
        os.makedirs(args.output_root, exist_ok=True)
        with open(os.path.join(args.output_root, 'vocab.txt'), 'w') as f:
            f.writelines(item + '\n' for item in vocabulary)

    for sc in scenes:
        scene_name = sc['name']
        print(f"== {scene_name} ==", flush=True)
        if not args.fixed_vocab:
            scene_vocab = load_txt(os.path.join(args.vocab_root, scene_name, 'scene_vocabulary.txt'))
            scene_vocab.append('sky')
            vocabulary = scene_vocab
        sample_token = sc['first_sample_token']
        kf = 0
        while sample_token:
            rec = nusc.get('sample', sample_token)
            for cam_name in CAMS:
                cam_sample = nusc.get('sample_data', rec['data'][cam_name])
                filename = cam_sample['filename']
                out_png = os.path.join(args.output_root, filename.replace('.jpg', '.png'))
                if os.path.exists(out_png):
                    continue
                os.makedirs(os.path.dirname(out_png), exist_ok=True)
                result = predictor.predict(os.path.join(args.data_root, filename), vocabulary=vocabulary)
                cv2.imwrite(out_png, result['sem_seg'])
            kf += 1
            if args.max_keyframes and kf >= args.max_keyframes:
                break
            sample_token = rec['next']
    print("SAN done", flush=True)


if __name__ == '__main__':
    main()
