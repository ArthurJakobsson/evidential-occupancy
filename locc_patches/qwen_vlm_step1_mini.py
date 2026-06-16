"""Step-1 LVLM vocabulary extraction, patched for nuScenes-mini on a 12 GB GPU.

Differences from qwen_vlm_step1.py:
  * uses Qwen-VL-Chat-Int4 (fits ~11 GB) instead of the fp16 model (~19 GB),
  * loads an explicit --version (default v1.0-mini) and iterates that version's scenes
    (the original ranged over numeric scene ids, which breaks for the scattered mini ids
    and would crash on non-mini gts dirs),
  * --scenes / --max_keyframes to run a quick subset for testing,
  * resumable (skips images whose .txt already exists).
Output: <output_root>/samples/<CAM>/<image>.txt  (one vocabulary paragraph per image).
"""
import os
from argparse import ArgumentParser

from nuscenes.nuscenes import NuScenes
from transformers import AutoModelForCausalLM, AutoTokenizer

CAMS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

TEXT1 = """
    This is an example image, where there exists these classes:
    traffic barrier; car; construction vehicle; crane;  pedestrian; traffic cone; road; sidewalk; terrain; grass; building; tree; fence; vegetaion; sky; traffic sign.
    Please carefully understand this image and these existing classes, and then describe the image in a brief paragraph.
"""
TEXT2 = """
    The image is captured by a camera on a driving car.
    Please carefully look at this image and detailedly describe the objects and background classes existed in this scene.
"""
TEXT3 = """
    Please list both the objects and background classes by a set of nouns.
    Organize them in a fixed format, each noun is separated by ';'.
    The format is similar to the nouns listed from the initial example image: traffic barrier; car; construction vehicle; crane;  pedestrian; traffic cone; road; sidewalk; terrain; grass; building; tree; fence; vegetaion; sky; traffic sign
"""


def parse_config():
    p = ArgumentParser()
    p.add_argument('--data_root', default='data/occ3d')
    p.add_argument('--output_root', default='data/occ3d/qwen_texts_step1')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--model', default='Qwen/Qwen-VL-Chat-Int4')
    p.add_argument('--scenes', default='', help='comma-separated scene names; default = all in version')
    p.add_argument('--max_keyframes', type=int, default=0, help='0 = all; else cap key-frames per scene (quick test)')
    p.add_argument('--example', default=os.path.join(os.path.dirname(__file__), 'example.jpg'))
    return p.parse_args()


def main():
    args = parse_config()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", trust_remote_code=True).eval()

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    scenes = list(nusc.scene)
    if args.scenes:
        want = {s.strip() for s in args.scenes.split(',') if s.strip()}
        scenes = [s for s in scenes if s['name'] in want]
    print(f"processing {len(scenes)} scenes (version {args.version}, model {args.model})", flush=True)

    for sc in scenes:
        print(f"== {sc['name']} ==", flush=True)
        sample_token = sc['first_sample_token']
        kf = 0
        while sample_token:
            rec = nusc.get('sample', sample_token)
            for cam_name in CAMS:
                cam_sample = nusc.get('sample_data', rec['data'][cam_name])
                filename = cam_sample['filename']
                out_txt = os.path.join(args.output_root, filename.replace('.jpg', '.txt'))
                if os.path.exists(out_txt):
                    continue
                img_path = os.path.join(args.data_root, filename)
                q1 = tokenizer.from_list_format([{'image': args.example}, {'text': TEXT1}])
                _, history = model.chat(tokenizer, query=q1, history=None)
                q2 = tokenizer.from_list_format([{'image': img_path}, {'text': TEXT2}])
                _, history = model.chat(tokenizer, query=q2, history=history)
                q3 = tokenizer.from_list_format([{'image': img_path}, {'text': TEXT3}])
                response, history = model.chat(tokenizer, query=q3, history=history)
                os.makedirs(os.path.dirname(out_txt), exist_ok=True)
                with open(out_txt, 'w') as f:
                    f.write(response)
            kf += 1
            if args.max_keyframes and kf >= args.max_keyframes:
                break
            sample_token = rec['next']
    print("step1 done", flush=True)


if __name__ == '__main__':
    main()
