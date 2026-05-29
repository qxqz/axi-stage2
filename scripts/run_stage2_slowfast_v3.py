"""
Stage 2 v3: SlowFast + Frame Label A/B bbox + taxonomy 묘사
- Slow: A/B bbox 있는 프레임 우선 선택 + bbox 표시
- Fast: 균등 샘플링 + bbox 표시
- 출력: taxonomy 용어 사용한 자유 묘사 (Stage 3 분류용)

--mode debug : GT object/place 사용
--mode infer : Stage 1 predictions.json top-3 place 사용
"""
import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

REPO_ROOT  = Path('/data/cyclamen/repos/car-accident-analysis')
MODEL_PATH = '/data/cyclamen/pretrained_models/Qwen2.5-VL-7B-Instruct'
NAS_LABEL  = Path('/data/cyclamen/repos/blackbox-analysis/data/095.교통사고_영상_데이터/01.데이터')

sys.path.append(str(REPO_ROOT))
from data.hierarchy_utils import HierarchyNavigator

with open(REPO_ROOT / 'config/mapping.json', encoding='utf-8') as f:
    mapping = json.load(f)

navigator = HierarchyNavigator(
    hierarchy_path=str(REPO_ROOT / 'config/hierarchy.json'),
    mapping_path=str(REPO_ROOT / 'config/mapping.json'),
)

CATEGORY_KO = {
    'vehicle':             '차량',
    'two-wheeled-vehicle': '이륜차',
    'bike':                '자전거',
    'pedestrian':          '보행자',
}


# ── A/B 카테고리 + POV 탐색 ──────────────────────────────
def get_actor_info(video_name, total_frames, split='val'):
    """
    frame label에서:
    - A/B 객체 카테고리 (차량/보행자/이륜차/자전거)
    - POV: 카메라가 A/B 중 어느 쪽인지
    """
    split_dir = '1.Training' if split == 'train' else '2.Validation'
    label_dir = NAS_LABEL / split_dir / '이미지라벨_extracted_correct' / video_name

    a_cat, b_cat = None, None
    pov = 'unknown'  # 카메라 차량

    for idx in range(min(total_frames, 150)):
        lf = label_dir / f'frame_{idx+1:05d}.json'
        if not lf.exists():
            continue
        with open(lf) as f:
            objs = json.load(f).get('objects', [])
        for obj in objs:
            cat = CATEGORY_KO.get(obj.get('category', ''), obj.get('category', ''))
            if obj.get('isObjectA') and not a_cat:
                a_cat = cat
                # A가 화면에 보인다 = 카메라는 B
                pov = 'B'
            if obj.get('isObjectB') and not b_cat:
                b_cat = cat
                # B가 화면에 보인다 = 카메라는 A (A가 먼저 발견되지 않았을 때만)
                if pov == 'unknown':
                    pov = 'A'
        if a_cat and b_cat:
            break

    return a_cat or 'A객체', b_cat or 'B객체', pov


# ── bbox 표시 ────────────────────────────────────────────
def annotate_frame(img, frame_idx, video_name, split='val'):
    """프레임에 A(빨강)/B(파랑) bbox + 카테고리 그리기"""
    split_dir = '1.Training' if split == 'train' else '2.Validation'
    lf = NAS_LABEL / split_dir / '이미지라벨_extracted_correct' \
         / video_name / f'frame_{frame_idx+1:05d}.json'
    if not lf.exists():
        return img

    with open(lf) as f:
        objs = json.load(f).get('objects', [])

    img = img.copy()
    draw = ImageDraw.Draw(img)
    for obj in objs:
        bbox = obj.get('bbox', [])
        if len(bbox) < 4:
            continue
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cat = CATEGORY_KO.get(obj.get('category', ''), obj.get('category', ''))
        if obj.get('isObjectA'):
            color, tag = 'red', f'A({cat})'
        elif obj.get('isObjectB'):
            color, tag = 'blue', f'B({cat})'
        else:
            continue
        draw.rectangle([x, y, x+w, y+h], outline=color, width=4)
        draw.rectangle([x, max(0, y-22), x+len(tag)*8, y], fill=color)
        draw.text((x+3, max(0, y-20)), tag, fill='white')
    return img


# ── 객체 프레임 탐색 ─────────────────────────────────────
def find_object_frames(video_name, total_frames, split='val'):
    split_dir = '1.Training' if split == 'train' else '2.Validation'
    label_dir = NAS_LABEL / split_dir / '이미지라벨_extracted_correct' / video_name

    a_only, b_only, ab_both = [], [], []
    for idx in range(min(total_frames, 150)):
        lf = label_dir / f'frame_{idx+1:05d}.json'
        if not lf.exists():
            continue
        with open(lf) as f:
            objs = json.load(f).get('objects', [])
        has_a = any(o.get('isObjectA') for o in objs)
        has_b = any(o.get('isObjectB') for o in objs)
        if has_a and has_b:
            ab_both.append(idx)
        elif has_a:
            a_only.append(idx)
        elif has_b:
            b_only.append(idx)

    return a_only, b_only, ab_both


# ── 프레임 로드 ──────────────────────────────────────────
def load_slowfast_frames(video_name, slow_n=4, fast_n=8, split='val'):
    """
    Slow: A/B bbox 있는 프레임 우선 선택 (~256토큰/프레임)
    Fast: 균등 샘플링 (~64토큰/프레임)
    둘 다 bbox 표시
    """
    path = REPO_ROOT / 'outputs' / 'sample_videos' / f'{video_name}.mp4'
    cap  = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], [], 30.0, 0.0

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fast_pixels = 64 * 28 * 28
    scale  = (fast_pixels / (W * H)) ** 0.5
    fast_W = max(28, int(W * scale // 28) * 28)
    fast_H = max(28, int(H * scale // 28) * 28)

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    a_only, b_only, ab_both = find_object_frames(video_name, total, split)
    obj_frames = ab_both + a_only + b_only

    if len(obj_frames) >= slow_n:
        sel = np.linspace(0, len(obj_frames)-1, slow_n, dtype=int)
        slow_idx = [obj_frames[i] for i in sel]
    else:
        fallback = np.linspace(0, total-1, slow_n, dtype=int).tolist()
        slow_idx = list(dict.fromkeys(obj_frames + fallback))[:slow_n]

    fast_idx = np.linspace(0, total-1, fast_n, dtype=int).tolist()

    slow = []
    for idx in slow_idx:
        f = read_frame(idx)
        if f:
            slow.append(annotate_frame(f, idx, video_name, split))

    fast = []
    for idx in fast_idx:
        f = read_frame(idx)
        if f:
            ann = annotate_frame(f, idx, video_name, split)
            fast.append(ann.resize((fast_W, fast_H), Image.LANCZOS))

    cap.release()
    return slow, fast, fps, duration


# ── 프롬프트 ────────────────────────────────────────────
def build_prompt(object_name, a_name, b_name, pov,
                 fps, duration, slow_n, fast_n,
                 object_id, place_ids, filming_way, pov_type):

    # POV 설명
    if filming_way == 'cc' or pov_type == 3:
        pov_desc = f"이 영상은 3인칭 시점입니다. 빨간 박스가 A({a_name}), 파란 박스가 B({b_name})입니다."
    elif pov == 'A':
        pov_desc = (f"이 영상은 A({a_name})에 장착된 블랙박스입니다. "
                    f"화면에 보이는 상대방 파란 박스가 B({b_name})입니다.")
    elif pov == 'B':
        pov_desc = (f"이 영상은 B({b_name})에 장착된 블랙박스입니다. "
                    f"화면에 보이는 상대방 빨간 박스가 A({a_name})입니다.")
    else:
        pov_desc = (f"이 영상은 블랙박스로 촬영되었습니다. "
                    f"빨간 박스가 A({a_name}), 파란 박스가 B({b_name})입니다.")

    # taxonomy 용어 수집
    combos = {}
    for place_id in place_ids:
        place_name = mapping['place'][str(place_id)]
        for fid in navigator.get_valid_features(int(object_id), int(place_id)):
            feat_name = navigator.reverse_mapping['feature'][fid]
            for a_id, b_id in navigator.get_valid_progress_pairs(
                    int(object_id), int(place_id), fid):
                a_t = navigator.reverse_mapping['a_progress'][a_id]
                b_t = navigator.reverse_mapping['b_progress'][b_id]
                combos.setdefault(feat_name, set()).add((a_t, b_t, place_name))

    vocab_str = ""
    for feat_name, pairs in combos.items():
        a_terms = sorted(set(a for a, b, _ in pairs))
        b_terms = sorted(set(b for a, b, _ in pairs))
        vocab_str += f"\n[{feat_name}]\n"
        vocab_str += f"  A({a_name}) 행동 후보: {', '.join(a_terms)}\n"
        vocab_str += f"  B({b_name}) 행동 후보: {', '.join(b_terms)}\n"

    place_str = ", ".join(mapping['place'][str(p)] for p in place_ids)

    return f"""{pov_desc}
사고 유형: {object_name} / 장소 후보: {place_str}
(앞 {slow_n}장 고해상도, 뒤 {fast_n}장 저해상도 전체 흐름 {duration:.1f}초)

아래는 이 사고에서 사용 가능한 분류 용어입니다. 반드시 이 용어들을 활용하여 서술하세요.
{vocab_str}
관찰된 내용을 바탕으로 3~4문장으로 서술하세요:
- 충돌 전 A({a_name})와 B({b_name}) 각각의 행동 (위 용어 사용)
- 충돌 방식과 상황
- 관찰되지 않은 내용은 서술하지 마세요."""


# ── 추론 ────────────────────────────────────────────────
def generate(model, processor, slow_frames, fast_frames,
             prompt, max_new_tokens=512):
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": f} for f in slow_frames]
                 + [{"type": "image", "image": f} for f in fast_frames]
                 + [{"type": "text",  "text": prompt}],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = out_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ── 메인 ────────────────────────────────────────────────
def main(args):
    print(f"Mode: {args.mode} | Model: {MODEL_PATH}")

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=quant_cfg,
        device_map="auto", low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=16  * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    model.eval()
    print(f"Model loaded | slow={args.slow_n}x~256tok | fast={args.fast_n}x~64tok")

    pred_db = {}
    if args.mode == 'infer':
        assert args.predictions, "--predictions 필요 (infer 모드)"
        with open(args.predictions) as f:
            pred_db = json.load(f)

    with open(args.fixed_samples) as f:
        video_names = json.load(f)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db = {}
    if out_path.exists():
        with open(out_path) as f:
            db = json.load(f)
        print(f"Resumed: {len(db)}")

    label_dir = REPO_ROOT / 'outputs' / 'sample_labels'

    for video_name in tqdm(video_names):
        if video_name in db:
            continue

        lf = label_dir / f'{video_name}.json'
        if not lf.exists():
            print(f"[SKIP] {video_name}: 라벨 없음")
            continue
        with open(lf) as f:
            gt = json.load(f)['video']

        if args.mode == 'debug':
            obj_id    = gt['accident_object']
            place_ids = [gt['accident_place']]
        else:
            if video_name not in pred_db:
                print(f"[SKIP] {video_name}: predictions 없음")
                continue
            p = pred_db[video_name]
            obj_id    = p['predicted_object']
            place_ids = p['top3_place_ids']

        obj_name     = mapping['object'][str(obj_id)]
        filming_way  = gt.get('filming_way', 'bb')
        pov_type     = gt.get('video_point_of_view', 1)

        slow_frames, fast_frames, fps, duration = load_slowfast_frames(
            video_name, args.slow_n, args.fast_n, args.split)
        if not slow_frames:
            print(f"[SKIP] {video_name}: 프레임 없음")
            continue

        # A/B 카테고리 + POV
        total = len(slow_frames) + len(fast_frames)
        a_name, b_name, pov = get_actor_info(video_name, 150, args.split)
        if filming_way == 'cc' or pov_type == 3:
            pov = 'third'

        prompt = build_prompt(
            obj_name, a_name, b_name, pov,
            fps, duration, len(slow_frames), len(fast_frames),
            str(obj_id), place_ids, filming_way, pov_type)

        try:
            response = generate(model, processor, slow_frames, fast_frames, prompt)
        except Exception as e:
            print(f"[ERROR] {video_name}: {e}")
            continue

        db[video_name] = {
            'object':      obj_id,
            'place_ids':   place_ids,
            'object_name': obj_name,
            'a_name':      a_name,
            'b_name':      b_name,
            'pov':         pov,
            'mode':        args.mode,
            'gt_feature':  mapping['feature'].get(
                               str(gt.get('accident_place_feature', '')), '?'),
            'gt_a':        mapping['a_progress'].get(
                               str(gt.get('vehicle_a_progress_info', '')), '?'),
            'gt_b':        mapping['b_progress'].get(
                               str(gt.get('vehicle_b_progress_info', '')), '?'),
            'response':    response,
        }

        if len(db) % 5 == 0:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(db, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint: {len(db)}")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print(f"Done. {len(db)} -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',    default='debug', choices=['debug', 'infer'])
    parser.add_argument('--split',   default='val',   choices=['train', 'val'])
    parser.add_argument('--fixed_samples', required=True)
    parser.add_argument('--predictions', default='')
    parser.add_argument('--output',  required=True)
    parser.add_argument('--slow_n',  type=int, default=4)
    parser.add_argument('--fast_n',  type=int, default=8)
    args = parser.parse_args()
    main(args)