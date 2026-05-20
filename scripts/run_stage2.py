"""
Stage 2: VLM 사고 설명 생성
  --use_gt      : Stage 1 추론 대신 GT 라벨 사용 (테스트용)
  --split       : train / val
  --stratify    : object 클래스별 균등 샘플링
  --n_per_class : stratify시 클래스당 샘플 수
  --num_samples : 일반 샘플링시 샘플 수 (0=전체)
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, argparse, random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from data.hierarchy_utils import HierarchyNavigator

REPO_ROOT       = Path('/data/cyclamen/repos/car-accident-analysis')
# MODEL_PATH      = '/data/cyclamen/pretrained_models/Qwen2.5-VL-72B-Instruct'
MODEL_PATH      = '/data/cyclamen/pretrained_models/Qwen2.5-VL-7B-Instruct'
FRAME_LABEL_DIR = Path('/data/cyclamen/repos/blackbox-analysis/data/095.교통사고_영상_데이터/01.데이터')

with open(REPO_ROOT / 'config/question_map.json', encoding='utf-8') as f:
    QUESTION_MAP = json.load(f)

QUESTION_TEXT = {
    1:  "신호등이 있었나? (있음/없음/한쪽만/unknown)",
    2:  "A의 신호 색깔은? (녹색/황색/적색/무신호/녹색좌회전/unknown)",
    3:  "B의 신호 색깔은? (녹색/황색/적색/무신호/녹색좌회전/unknown)",
    4:  "비보호좌회전 표지가 있었나? (있음/없음/unknown)",
    5:  "일시정지 표지 방향은? (A방향/B방향/없음/unknown)",
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
    27: "이륜차/자전거 위치는? (A앞/A옆/역방향/unknown)",
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
    40: "유턴구역 종류는? (상시유턴구역/신호유턴구역/unknown)",
    41: "유턴 선후 관계는? (A선행B급유턴/동시유턴선행후행/unknown)",
    42: "노면 표시 위반이 있었나? (A위반/B위반/없음/unknown)",
    43: "차량이 본선으로 합류 중이었나? (A합류/B합류/없음/unknown)",
    44: "주차구역에서 출자 방향은? (직진출자/후진출자/unknown)",
    45: "두 차량 회전 각도가 90도 미만이었나? (90도미만/90도이상/unknown)",
    46: "이륜차/자전거와 차량이 동일 차로에 나란히 있었나? (있음/없음/unknown)",
}


# ── POV 결정 ────────────────────────────────────────────
def get_pov_vehicle(split: str, video_name: str) -> str:
    """
    프레임 라벨에서 카메라 탑재 차량 결정
    isObjectA가 보이면 → 카메라는 B차량
    isObjectB가 보이면 → 카메라는 A차량
    """
    split_dir = '1.Training' if split == 'train' else '2.Validation'
    frame_dir = FRAME_LABEL_DIR / split_dir / '이미지라벨_extracted_correct' / video_name

    if not frame_dir.exists():
        return 'unknown'

    for frame_file in sorted(frame_dir.glob('*.json'))[:10]:
        with open(frame_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for obj in data.get('objects', []):
            if obj.get('isObjectA'):
                return 'B'
            if obj.get('isObjectB'):
                return 'A'

    return 'unknown'


# ── 프롬프트 ────────────────────────────────────────────
def build_prompt(object_name, place_name, pov_desc,
                 fps, duration, num_frames,
                 object_id, place_id):

    combo_key = f"{object_id}_{place_id}"
    q_nums = QUESTION_MAP.get(combo_key, {}).get('questions', [7, 8, 9])

    question_block = "\n".join(
        f"Q{q}. {QUESTION_TEXT[q]}" for q in q_nums
    )
    answer_format = "\n".join(f"Q{q}:" for q in q_nums)

    return f"""{pov_desc}
촬영 정보: 총 {duration:.1f}초 영상 ({fps:.0f}fps), {num_frames}프레임 균등 추출
사고 유형: {object_name} / 장소: {place_name}

아래 질문에 영상에서 관찰된 것만 답하세요.
관찰 불가능하거나 보이지 않으면 반드시 "unknown"으로 답하세요.
각 답변은 제시된 선택지 중 하나로만 답하세요.

{question_block}

답변:
{answer_format}"""


# ── 프레임 로드 ─────────────────────────────────────────
def load_frames(frames_dir: Path, video_name: str, num_frames: int) -> list:
    files = sorted(frames_dir.glob(f"{video_name}_frame_*.jpg"))
    if not files:
        return []
    idx = np.linspace(0, len(files) - 1, num_frames, dtype=int)
    return [Image.open(files[i]).convert('RGB') for i in idx]


# ── VLM 추론 ────────────────────────────────────────────
def generate_description(model, processor, frames: list,
                         prompt: str, max_new_tokens: int = 512) -> str:
    messages = [{
        "role": "system",
        "content": "당신은 교통사고 블랙박스 영상을 분석하는 전문가입니다. 영상에서 관찰된 사실만 간결하게 서술합니다.",
    }, {
        "role": "user",
        "content": [{"type": "image", "image": f} for f in frames]
                 + [{"type": "text",  "text": prompt}],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated = out_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ── 샘플링 ──────────────────────────────────────────────
def stratified_sample(label_dir: Path, n_per_class: int) -> dict:
    buckets = defaultdict(list)
    for lf in label_dir.glob('*.json'):
        with open(lf, 'r', encoding='utf-8') as f:
            gt = json.load(f)['video']
        buckets[gt['accident_object']].append(lf)

    samples = {}
    for obj_cls, files in sorted(buckets.items()):
        picked = random.sample(files, min(n_per_class, len(files)))
        for lf in picked:
            with open(lf, 'r', encoding='utf-8') as f:
                gt = json.load(f)['video']
            samples[lf.stem] = {
                'object':       gt['accident_object'],
                'place':        gt['accident_place'],
                'filming_way':  gt.get('filming_way', 'bb'),
            }

    print(f"[Stratified] {len(samples)} samples ({len(buckets)} classes × ~{n_per_class})")
    return samples


# ── 메인 ────────────────────────────────────────────────
def main(args):
    with open(REPO_ROOT / 'config/mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    navigator = HierarchyNavigator(
        hierarchy_path=str(REPO_ROOT / 'config/hierarchy.json'),
        mapping_path=str(REPO_ROOT / 'config/mapping.json'),
    )

    frames_dir = Path(f'/local_datasets/cyclamen/frames_uniform/{args.split}/frames')

    # ── 입력 소스 결정 ──
    if args.fixed_samples:
        with open(args.fixed_samples, 'r') as f:
            fixed_names = json.load(f)

        split_key = '1.Training' if args.split == 'train' else '2.Validation'
        label_dir = Path(f'/data/cyclamen/repos/blackbox-analysis/data/095.교통사고_영상_데이터/01.데이터/{split_key}/라벨_extracted')

        samples = {}
        for name in fixed_names:
            lf = label_dir / f"{name}.json"
            if not lf.exists():
                continue
            with open(lf, 'r', encoding='utf-8') as f:
                gt = json.load(f)['video']
            samples[name] = {
                'object':      gt['accident_object'],
                'place':       gt['accident_place'],
                'filming_way': gt.get('filming_way', 'bb'),
            }
        print(f"[Fixed samples] {len(samples)} samples")

    elif args.use_gt:
        split_key = '1.Training' if args.split == 'train' else '2.Validation'
        label_dir = Path(f'/data/cyclamen/repos/blackbox-analysis/data/095.교통사고_영상_데이터/01.데이터/{split_key}/라벨_extracted')

        if args.stratify:
            samples = stratified_sample(label_dir, args.n_per_class)
        else:
            label_files = sorted(label_dir.glob('*.json'))
            if args.num_samples:
                random.seed(42)
                label_files = random.sample(label_files, min(args.num_samples, len(label_files)))
            samples = {}
            for lf in label_files:
                with open(lf, 'r', encoding='utf-8') as f:
                    gt = json.load(f)['video']
                samples[lf.stem] = {
                    'object':      gt['accident_object'],
                    'place':       gt['accident_place'],
                    'filming_way': gt.get('filming_way', 'bb'),
                }
        print(f"[GT mode] {len(samples)} samples")

    else:
        with open(args.predictions, 'r', encoding='utf-8') as f:
            preds = json.load(f)
        # predictions.json에 filming_way 없으면 GT에서 보완
        split_key = '1.Training' if args.split == 'train' else '2.Validation'
        label_dir = Path(f'/data/cyclamen/repos/blackbox-analysis/data/095.교통사고_영상_데이터/01.데이터/{split_key}/라벨_extracted')

        samples = {}
        for k, v in preds.items():
            lf = label_dir / f"{k}.json"
            filming_way = 'bb'
            if lf.exists():
                with open(lf, 'r', encoding='utf-8') as f:
                    filming_way = json.load(f)['video'].get('filming_way', 'bb')
            samples[k] = {
                'object':      v['predicted_object'],
                'place':       v['predicted_place'],
                'filming_way': filming_way,
            }

        if args.num_samples:
            keys = list(samples.keys())[:args.num_samples]
            samples = {k: samples[k] for k in keys}
        print(f"[Stage1 predictions] {len(samples)} samples")

    # ── VLM 로드 ──
    print("Loading Qwen2.5-VL (4bit)...")
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
        min_pixels=64  * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    model.eval()

    # ── resume ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db = {}
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            db = json.load(f)
        print(f"Resumed: {len(db)} already done")

    # ── 생성 ──
    for video_name, info in tqdm(samples.items()):
        if video_name in db:
            continue

        frames = load_frames(frames_dir, video_name, args.num_frames)
        if not frames:
            print(f"[SKIP] {video_name}: 프레임 없음")
            continue

        obj_id   = info['object']
        place_id = info['place']
        filming_way = info.get('filming_way', 'bb')

        # video_point_of_view 읽기
        pov_type = 1  # 기본 1인칭
        lf = label_dir / f"{video_name}.json"
        if lf.exists():
            with open(lf, 'r', encoding='utf-8') as f:
                pov_type = json.load(f)['video'].get('video_point_of_view', 1)

        obj_name   = mapping['object'][str(obj_id)]
        place_name = mapping['place'][str(place_id)]

        # POV 결정
        if filming_way == 'cc' or pov_type == 3:
            pov = 'third_person'
        else:
            pov = get_pov_vehicle(args.split, video_name)

        # pov_desc 생성
        if pov == 'third_person':
            pov_desc = "이 영상은 3인칭 시점으로 촬영되었습니다. 화면에 A차량과 B차량이 모두 보입니다."
        elif pov == 'unknown':
            pov_desc = "이 영상은 블랙박스로 촬영되었습니다."
        else:
            ego   = f"{pov}차량"
            other = "B차량" if pov == 'A' else "A차량"
            pov_desc = (f"이 영상은 {ego}에 장착된 블랙박스로 촬영되었습니다. "
                        f"화면에 보이는 상대 차량이 {other}입니다.")
    
        import cv2

        # fps, duration 추출
        split_key = '1.Training' if args.split == 'train' else '2.Validation'
        video_path = Path(f'/data/cyclamen/repos/blackbox-analysis/data/095.교통사고_영상_데이터/01.데이터/{split_key}/영상_extracted/{video_name}.mp4')
        label_path = Path(f'/data/cyclamen/repos/blackbox-analysis/data/095.교통사고_영상_데이터/01.데이터/{split_key}/라벨_extracted/{video_name}.json')

        fps, duration = 30.0, 0.0  # fallback
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total / fps if fps > 0 else 0.0
            cap.release()

        prompt = build_prompt(
            obj_name, place_name, pov_desc,
            fps, duration, args.num_frames,
            obj_id, place_id,
        )

        try:
            desc = generate_description(model, processor, frames, prompt,
                                        max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"[ERROR] {video_name}: {e}")
            continue

        gt_info = {}
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                gt = json.load(f)['video']
            feat_id = str(gt.get('accident_place_feature', ''))
            a_id    = str(gt.get('vehicle_a_progress_info', ''))
            b_id    = str(gt.get('vehicle_b_progress_info', ''))
            gt_info = {
                'gt_feature': mapping['feature'].get(feat_id, f'?({feat_id})'),
                'gt_a':       mapping['a_progress'].get(a_id, f'?({a_id})'),
                'gt_b':       mapping['b_progress'].get(b_id, f'?({b_id})'),
            }

        db[video_name] = {
            'object':       obj_id,
            'place':        place_id,
            'object_name':  obj_name,
            'place_name':   place_name,
            'filming_way':  filming_way,
            'pov_vehicle':  pov,
            **gt_info,
            'qa_response':  desc,
        }

        if len(db) % 50 == 0:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(db, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint: {len(db)} saved")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(db)} descriptions → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',          default='val', choices=['train', 'val'])
    parser.add_argument('--use_gt',         action='store_true')
    parser.add_argument('--predictions',    default='')
    parser.add_argument('--output',         required=True)
    parser.add_argument('--stratify',       action='store_true')
    parser.add_argument('--n_per_class',    type=int, default=3)
    parser.add_argument('--num_samples',    type=int, default=0)
    parser.add_argument('--num_frames',     type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--fixed_samples', default='', help='고정 샘플 리스트 JSON 경로')
    args = parser.parse_args()

    if not args.use_gt and not args.predictions:
        parser.error("--use_gt 없으면 --predictions 필요")

    main(args)