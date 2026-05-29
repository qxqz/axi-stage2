"""
Stage 2 Q&A v1: SlowFast + Q&A 방식
- 핵심 질문(place-independent)은 항상 포함
- top-3 place 각각의 질문 union으로 추가
- A/B bbox 표시 프레임 사용

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

with open(REPO_ROOT / 'config/mapping.json', encoding='utf-8') as f:
    mapping = json.load(f)
with open(REPO_ROOT / 'config/question_map.json', encoding='utf-8') as f:
    QUESTION_MAP = json.load(f)

CATEGORY_KO = {
    'vehicle':             '차량',
    'two-wheeled-vehicle': '이륜차',
    'bike':                '자전거',
    'pedestrian':          '보행자',
}

# 항상 묻는 핵심 질문 (place-independent)
CORE_QUESTIONS = {7, 8, 9, 11, 14}

QUESTION_TEXT = {
    1:  "신호등이 있었나? (있음/없음/한쪽만/unknown)",
    2:  "A의 신호 색깔은? (녹색/황색/적색/무신호/녹색좌회전/unknown)",
    3:  "B의 신호 색깔은? (녹색/황색/적색/무신호/녹색좌회전/unknown)",
    4:  "비보호좌회전 표지가 있었나? (있음/없음/unknown)",
    5:  "일시정지 표지 방향은? (A방향/B방향/없음/해당없음/unknown)",
    6:  "일방통행 위반은? (A위반/B위반/없음/unknown)",
    7:  "A의 진행 방향은? (직진/좌회전/우회전/유턴/정차/후진/unknown)",
    8:  "B의 진행 방향은? (직진/좌회전/우회전/유턴/정차/후진/unknown)",
    9:  "두 객체의 상대 방향은? (같은방향/교차/대향/unknown)",
    10: "교차로 진입 순서는? (A먼저/B먼저/동시/해당없음/unknown)",
    11: "차로 변경한 쪽은? (A/B/없음/unknown)",
    12: "추월 시도한 쪽은? (A/B/없음/unknown)",
    13: "중앙선 침범한 쪽은? (A/B/없음/unknown)",
    14: "선후 관계는? (A선행/B선행/나란히/해당없음/unknown)",
    15: "대로소로 구분은? (A가대로/A가소로/동일폭/unknown)",
    16: "안전지대 있었나? (있음/없음/unknown)",
    17: "차로 폭은? (나란히가능/단일/2차로이상/unknown)",
    18: "정차 차량은? (A정차/B정차/없음/unknown)",
    19: "열린 문은? (A문열림/B문열림/없음/unknown)",
    20: "역주행은? (A역주행/B역주행/없음/unknown)",
    21: "긴급자동차는? (A긴급/B긴급/없음/unknown)",
    22: "낙하물은? (있음/없음/unknown)",
    23: "횡단 중인 보행자/이륜차/자전거가 있었나? (있음/없음/unknown)",
    24: "보행자 신호 색깔은? (녹색/녹색점멸/적색/없음/unknown)",
    25: "횡단보도와의 거리는? (위/10m이내/10m이상/unknown)",
    26: "이륜차/자전거 차도 진입은? (차도외에서진입/차도에서이탈/없음/unknown)",
    28: "자전거도로 종류는? (전용도로/전용차로/우선도로/unknown)",
    29: "회전교차로 차로 수는? (1차로형/2차로형/unknown)",
    30: "회전교차로에서 각 차량 상태는? (A진입중B회전중/A회전중B진입중/둘다진입중/교차로내진로변경/unknown)",
    31: "자동차가 횡단보도를 통과했나? (통과전/통과후/unknown)",
    32: "보행자 신호가 점멸이었나? (녹색점멸/적색점멸/일반/unknown)",
    33: "보행자/자전거가 자전거횡단도로 위에 있었나? (자전거횡단도로위/일반도로위/unknown)",
    34: "우회전 반경은? (대우회전/소우회전/unknown)",
    35: "고속도로 보행이 사무/공무/고장 목적이었나? (이유없음/사무공무고장/unknown)",
    36: "차량이 후진 중이었나? (A후진/B후진/없음/unknown)",
    37: "공사/장애물로 인한 불가피한 보행 상황이었나? (있음/없음/unknown)",
    38: "안전지대를 이미 벗어난 후 사고가 났나? (벗어나기전/벗어난후/unknown)",
    39: "정체 차로 상황이었나? (있음/없음/unknown)",
    40: "유턴구역 종류는? (상시유턴구역/신호유턴구역/해당없음/unknown)",
    41: "유턴 선후 관계는? (A선행B급유턴/동시유턴선행후행/해당없음/unknown)",
    42: "노면 표시 위반이 있었나? (A위반/B위반/없음/unknown)",
    43: "차량이 본선으로 합류 중이었나? (A합류/B합류/없음/unknown)",
    44: "주차구역에서 출자 방향은? (직진출자/후진출자/unknown)",
    45: "두 차량 회전 각도가 90도 미만이었나? (90도미만/90도이상/unknown)",
    46: "이륜차/자전거와 차량이 동일 차로에 나란히 있었나? (있음/없음/unknown)",
    47: "중앙선 종류는? (실선/점선/없음/unknown)",
}


# ── A/B 카테고리 + POV 탐색 ──────────────────────────────
def get_actor_info(video_name, total_frames, split='val'):
    split_dir = '1.Training' if split == 'train' else '2.Validation'
    label_dir = NAS_LABEL / split_dir / '이미지라벨_extracted_correct' / video_name

    a_cat, b_cat = None, None
    pov = 'unknown'

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
                pov = 'B'
            if obj.get('isObjectB') and not b_cat:
                b_cat = cat
                if pov == 'unknown':
                    pov = 'A'
        if a_cat and b_cat:
            break

    return a_cat or 'A객체', b_cat or 'B객체', pov


# ── bbox 표시 ────────────────────────────────────────────
def annotate_frame(img, frame_idx, video_name, split='val'):
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

    slow, fast = [], []
    for idx in slow_idx:
        f = read_frame(idx)
        if f:
            slow.append(annotate_frame(f, idx, video_name, split))
    for idx in fast_idx:
        f = read_frame(idx)
        if f:
            ann = annotate_frame(f, idx, video_name, split)
            fast.append(ann.resize((fast_W, fast_H), Image.LANCZOS))

    cap.release()
    return slow, fast, fps, duration


# ── 질문 세트 구성 ───────────────────────────────────────
def get_question_set(object_id, place_ids):
    """핵심 질문 + top-3 place 유니온"""
    q_set = set(CORE_QUESTIONS)
    for place_id in place_ids:
        combo_key = f"{object_id}_{place_id}"
        qs = QUESTION_MAP.get(combo_key, {}).get('questions', [])
        q_set.update(qs)
    return sorted(q_set)


# ── 프롬프트 ────────────────────────────────────────────
def build_prompt(object_name, a_name, b_name, pov,
                 fps, duration, slow_n, fast_n,
                 object_id, place_ids, filming_way, pov_type):

    # POV 설명
    if filming_way == 'cc' or pov_type == 3:
        pov_desc = (f"이 영상은 3인칭 시점입니다. "
                    f"빨간 박스가 A({a_name}), 파란 박스가 B({b_name})입니다.")
    elif pov == 'A':
        pov_desc = (f"이 영상은 A({a_name})에 장착된 블랙박스입니다. "
                    f"화면에 보이는 상대방 파란 박스가 B({b_name})입니다.")
    elif pov == 'B':
        pov_desc = (f"이 영상은 B({b_name})에 장착된 블랙박스입니다. "
                    f"화면에 보이는 상대방 빨간 박스가 A({a_name})입니다.")
    else:
        pov_desc = (f"이 영상은 블랙박스로 촬영되었습니다. "
                    f"빨간 박스가 A({a_name}), 파란 박스가 B({b_name})입니다.")

    # 질문 세트 구성
    q_nums = get_question_set(str(object_id), place_ids)
    # A/B 명칭 치환
    question_block = "\n".join(
        f"Q{q}. {QUESTION_TEXT[q].replace('A의', f'A({a_name})의').replace('B의', f'B({b_name})의')}"
        for q in q_nums if q in QUESTION_TEXT
    )
    answer_format = "\n".join(f"Q{q}:" for q in q_nums if q in QUESTION_TEXT)

    place_str = ", ".join(mapping['place'][str(p)] for p in place_ids)

    return f"""{pov_desc}
사고 유형: {object_name} / 장소 후보: {place_str}
(앞 {slow_n}장 고해상도 ~256토큰/프레임, 뒤 {fast_n}장 저해상도 ~64토큰/프레임, 총 {duration:.1f}초)

아래 질문에 영상에서 관찰된 것만 답하세요.
관찰 불가능하거나 보이지 않으면 반드시 "unknown"으로 답하세요.
각 답변은 제시된 선택지 중 하나로만 답하세요.

{question_block}

답변:
{answer_format}"""


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

        obj_name    = mapping['object'][str(obj_id)]
        filming_way = gt.get('filming_way', 'bb')
        pov_type    = gt.get('video_point_of_view', 1)

        slow_frames, fast_frames, fps, duration = load_slowfast_frames(
            video_name, args.slow_n, args.fast_n, args.split)
        if not slow_frames:
            print(f"[SKIP] {video_name}: 프레임 없음")
            continue

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