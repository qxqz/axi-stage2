"""
계층 구조 네비게이터

상위 분류 결과로 하위 후보를 필터링하는 유틸리티
"""
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class HierarchyNavigator:
    """계층 구조 탐색 및 필터링"""
    
    def __init__(self, hierarchy_path: str, mapping_path: str):
        """
        Args:
            hierarchy_path: 계층 구조 JSON 경로
            mapping_path: 텍스트-숫자 매핑 JSON 경로
        """
        # 계층 구조 로드
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            self.hierarchy = json.load(f)
        
        # 매핑 로드
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
        
        # 역매핑 (숫자 -> 텍스트)
        self.reverse_mapping = {
            category: {int(k): v for k, v in items.items()}
            for category, items in self.mapping.items()
        }
        
        print("[HierarchyNavigator] Initialized")
        self._print_stats()
    
    def _print_stats(self):
        """통계 출력"""
        # 객체별 장소 수
        for obj_name, places in self.hierarchy.items():
            print(f"  - {obj_name}: {len(places)} places")
    
    def get_valid_places(self, object_id: int) -> List[int]:
        """
        객체 ID로 가능한 장소들 반환
        
        Args:
            object_id: 사고 객체 ID (0-3)
        
        Returns:
            valid_place_ids: 가능한 장소 ID 리스트
        """
        object_name = self.reverse_mapping['object'].get(object_id)
        if not object_name:
            return []
        
        if object_name not in self.hierarchy:
            return []
        
        places = self.hierarchy[object_name].keys()
        
        valid_place_ids = []
        for place_id, place_name in self.mapping['place'].items():
            if place_name in places:
                valid_place_ids.append(int(place_id))
        
        return sorted(valid_place_ids)
    
    def get_valid_features(self, object_id: int, place_id: int) -> List[int]:
        """
        객체+장소로 가능한 특징들 반환
        
        Args:
            object_id: 사고 객체 ID
            place_id: 사고 장소 ID
        
        Returns:
            valid_feature_ids: 가능한 특징 ID 리스트
        """
        object_name = self.reverse_mapping['object'].get(object_id)
        place_name = self.reverse_mapping['place'].get(place_id)
        
        if not object_name or not place_name:
            return []
        
        try:
            features = self.hierarchy[object_name][place_name].keys()
            
            valid_feature_ids = []
            for feat_id, feat_name in self.mapping['feature'].items():
                if feat_name in features:
                    valid_feature_ids.append(int(feat_id))
            
            return sorted(valid_feature_ids)
        
        except KeyError:
            return []
    
    def get_valid_progress_pairs(
        self, 
        object_id: int, 
        place_id: int, 
        feature_id: int
    ) -> List[Tuple[int, int]]:
        """
        객체+장소+특징으로 가능한 (A, B) 진행 정보 조합들 반환
        
        Args:
            object_id: 사고 객체 ID
            place_id: 사고 장소 ID
            feature_id: 사고 특징 ID
        
        Returns:
            valid_pairs: [(a_id, b_id), ...] 리스트
        """
        object_name = self.reverse_mapping['object'].get(object_id)
        place_name = self.reverse_mapping['place'].get(place_id)
        feature_name = self.reverse_mapping['feature'].get(feature_id)
        
        if not all([object_name, place_name, feature_name]):
            return []
        
        try:
            progress_pairs = self.hierarchy[object_name][place_name][feature_name]
            
            valid_pairs = []
            for ab_pair_key, info in progress_pairs.items():
                a_text = info['A']
                b_text = info['B']
                
                # 텍스트 -> ID 변환
                a_id = None
                for aid, aname in self.mapping['a_progress'].items():
                    if aname == a_text:
                        a_id = int(aid)
                        break
                
                b_id = None
                for bid, bname in self.mapping['b_progress'].items():
                    if bname == b_text:
                        b_id = int(bid)
                        break
                
                if a_id is not None and b_id is not None:
                    valid_pairs.append((a_id, b_id))
            
            return valid_pairs
        
        except KeyError:
            return []
    
    def get_negligence(
        self,
        object_id: int,
        place_id: int,
        feature_id: int,
        a_progress_id: int,
        b_progress_id: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        예측된 조합의 과실 비율 가져오기
        
        Args:
            object_id: 사고 객체 ID
            place_id: 사고 장소 ID
            feature_id: 사고 특징 ID
            a_progress_id: A차량 진행 정보 ID
            b_progress_id: B차량 진행 정보 ID
        
        Returns:
            (negligence_A, negligence_B) 또는 (None, None)
        """
        # ID -> 텍스트 변환
        object_text = self.reverse_mapping['object'].get(object_id)
        place_text = self.reverse_mapping['place'].get(place_id)
        feature_text = self.reverse_mapping['feature'].get(feature_id)
        a_text = self.reverse_mapping['a_progress'].get(a_progress_id)
        b_text = self.reverse_mapping['b_progress'].get(b_progress_id)
        
        if not all([object_text, place_text, feature_text, a_text, b_text]):
            return None, None
        
        # 계층 구조에서 찾기
        try:
            progress_pairs = self.hierarchy[object_text][place_text][feature_text]
            
            # A|B 키 찾기
            for ab_key, info in progress_pairs.items():
                if info['A'] == a_text and info['B'] == b_text:
                    return info['negligence_A'], info['negligence_B']
        
        except KeyError:
            pass
        
        return None, None
    
    def validate_path(
        self,
        object_id: int,
        place_id: int,
        feature_id: int,
        a_progress_id: int,
        b_progress_id: int,
    ) -> bool:
        """
        예측 경로가 계층 구조상 유효한지 검증
        
        Returns:
            True if valid, False otherwise
        """
        # 1. 장소 검증
        valid_places = self.get_valid_places(object_id)
        if place_id not in valid_places:
            return False
        
        # 2. 특징 검증
        valid_features = self.get_valid_features(object_id, place_id)
        if feature_id not in valid_features:
            return False
        
        # 3. A/B 진행 정보 검증
        valid_pairs = self.get_valid_progress_pairs(object_id, place_id, feature_id)
        if (a_progress_id, b_progress_id) not in valid_pairs:
            return False
        
        return True
    
    def get_summary(
        self,
        object_id: int,
        place_id: int = None,
        feature_id: int = None,
    ) -> str:
        """
        현재 경로의 요약 문자열 생성
        
        Returns:
            summary: "차대차 > 직선도로 > 추돌사고"
        """
        parts = []
        
        obj_text = self.reverse_mapping['object'].get(object_id, f"Object{object_id}")
        parts.append(obj_text)
        
        if place_id is not None:
            place_text = self.reverse_mapping['place'].get(place_id, f"Place{place_id}")
            parts.append(place_text)
        
        if feature_id is not None:
            feat_text = self.reverse_mapping['feature'].get(feature_id, f"Feature{feature_id}")
            parts.append(feat_text)
        
        return " > ".join(parts)


if __name__ == '__main__':
    # 테스트
    nav = HierarchyNavigator(
        hierarchy_path='../config/hierarchy.json',
        mapping_path='../config/mapping.json'
    )
    
    # 차대차 (0)의 가능한 장소
    places = nav.get_valid_places(0)
    print(f"\n차대차 가능한 장소: {places}")
    
    # 차대차 + 직선도로의 가능한 특징
    features = nav.get_valid_features(0, 0)
    print(f"차대차 + 직선도로 가능한 특징: {features}")
    
    # 전체 경로 검증
    is_valid = nav.validate_path(0, 0, 0, 0, 1)
    print(f"경로 유효성: {is_valid}")