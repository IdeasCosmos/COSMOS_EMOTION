"""
COSMOS EMOTION - 양방향 계층 전파 시스템
==========================================

Part 2: Bidirectional Hierarchical Propagation System
- 5개 계층: MORPHEME → WORD → PHRASE → SENTENCE → DISCOURSE
- 양방향 전파: Bottom-Up (0.7배) + Top-Down (0.9^depth)
- 동적 가중치 조정
- 계층 간 상호작용 모델링

🔥 시스템 핵심 개념:

1. **상향 전파 (Bottom-Up)**:
   - 하층의 신호가 상층으로 전달
   - 매 층마다 70%만 전달 (0.7배)
   - 노이즈 필터링 효과
   
2. **하향 전파 (Top-Down)**:
   - 상층의 맥락이 하층으로 영향
   - 깊이에 따라 지수 감쇠 (0.9^depth)
   - 전역 문맥 반영

3. **양방향 균형**:
   - 두 방향의 신호를 통합
   - 상호작용 강도 계산
   - 지배적 방향 파악
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import copy


# ============================================================================
# 계층 정의
# ============================================================================

class Layer(Enum):
    """
    5개 계층 정의
    
    음악적 비유:
    - MORPHEME: 개별 음표 (♩, ♪, ♬)
    - WORD: 기본 코드 (C, Am, G7)
    - PHRASE: 악구/마디 (musical phrase)
    - SENTENCE: 악절 (period)
    - DISCOURSE: 악장 (movement)
    """
    MORPHEME = 1   # 형태소 - 가장 미시적
    WORD = 2       # 단어 - HIT 시스템 작동 지점
    PHRASE = 3     # 구 - 의미 단위
    SENTENCE = 4   # 문장 - 완결된 생각
    DISCOURSE = 5  # 담화 - 전체 맥락


@dataclass
class LayerConfig:
    """계층별 설정"""
    level: int                    # 계층 레벨 (1~5)
    name: str                     # 계층 이름
    up_transmission_ratio: float  # 상향 전달 비율
    down_decay_base: float        # 하향 감쇠 기저
    default_weight: float         # 기본 가중치
    confidence_threshold: float   # 신뢰도 임계값


# 계층별 설정값
LAYER_CONFIGS = {
    Layer.MORPHEME: LayerConfig(
        level=1,
        name="Morpheme",
        up_transmission_ratio=0.7,
        down_decay_base=0.9,
        default_weight=0.15,
        confidence_threshold=0.3
    ),
    Layer.WORD: LayerConfig(
        level=2,
        name="Word",
        up_transmission_ratio=0.7,
        down_decay_base=0.9,
        default_weight=0.25,  # HIT 시스템의 주력
        confidence_threshold=0.5
    ),
    Layer.PHRASE: LayerConfig(
        level=3,
        name="Phrase",
        up_transmission_ratio=0.7,
        down_decay_base=0.9,
        default_weight=0.25,
        confidence_threshold=0.4
    ),
    Layer.SENTENCE: LayerConfig(
        level=4,
        name="Sentence",
        up_transmission_ratio=0.7,
        down_decay_base=0.9,
        default_weight=0.20,
        confidence_threshold=0.6
    ),
    Layer.DISCOURSE: LayerConfig(
        level=5,
        name="Discourse",
        up_transmission_ratio=0.7,
        down_decay_base=0.9,
        default_weight=0.15,
        confidence_threshold=0.7
    )
}


# ============================================================================
# 감정 벡터 (28차원)
# ============================================================================

@dataclass
class EmotionVector:
    """
    28차원 감정 벡터
    - 기본 감정 7개 + 한국 감정 5개 + 기타 16개
    """
    # 기본 감정 7개
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    disgust: float = 0.0
    surprise: float = 0.0
    neutral: float = 0.5  # 중립은 기준선
    
    # 한국 특유 감정 5개
    han: float = 0.0        # 한 (恨)
    jeong: float = 0.0      # 정 (情)
    nunchi: float = 0.0     # 눈치
    hyeontta: float = 0.0   # 혀따닥 (짜증)
    menboong: float = 0.0   # 멘붕
    
    # 기타 감정 16개 (필요시 확장)
    excitement: float = 0.0
    calmness: float = 0.0
    empathic_pain: float = 0.0
    amusement: float = 0.0
    confusion: float = 0.0
    disappointment: float = 0.0
    guilt: float = 0.0
    shame: float = 0.0
    pride: float = 0.0
    relief: float = 0.0
    hope: float = 0.0
    despair: float = 0.0
    nostalgia: float = 0.0
    contempt: float = 0.0
    envy: float = 0.0
    gratitude: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """딕셔너리로 변환"""
        return {
            'joy': self.joy,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'disgust': self.disgust,
            'surprise': self.surprise,
            'neutral': self.neutral,
            'han': self.han,
            'jeong': self.jeong,
            'nunchi': self.nunchi,
            'hyeontta': self.hyeontta,
            'menboong': self.menboong,
            'excitement': self.excitement,
            'calmness': self.calmness,
            'empathic_pain': self.empathic_pain,
            'amusement': self.amusement,
            'confusion': self.confusion,
            'disappointment': self.disappointment,
            'guilt': self.guilt,
            'shame': self.shame,
            'pride': self.pride,
            'relief': self.relief,
            'hope': self.hope,
            'despair': self.despair,
            'nostalgia': self.nostalgia,
            'contempt': self.contempt,
            'envy': self.envy,
            'gratitude': self.gratitude,
        }
    
    def to_array(self) -> np.ndarray:
        """NumPy 배열로 변환"""
        return np.array(list(self.to_dict().values()))
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'EmotionVector':
        """딕셔너리에서 생성"""
        return cls(**d)
    
    def normalize_excluding_neutral(self) -> 'EmotionVector':
        """
        중립 제외하고 정규화
        - 중립은 기준선이므로 합산에서 제외!
        """
        d = self.to_dict()
        neutral_value = d.pop('neutral')
        
        total = sum(d.values())
        if total > 0:
            normalized = {k: v / total for k, v in d.items()}
        else:
            normalized = d
        
        normalized['neutral'] = neutral_value
        return EmotionVector.from_dict(normalized)
    
    def intensity(self) -> float:
        """
        감정 강도 (중립 제외)
        """
        d = self.to_dict()
        d.pop('neutral')
        return sum(d.values())
    
    def copy(self) -> 'EmotionVector':
        """복사본 생성"""
        return EmotionVector.from_dict(self.to_dict())


@dataclass
class LayerEmotionState:
    """
    각 계층의 감정 상태
    """
    layer: Layer
    emotion_vector: EmotionVector
    confidence: float = 0.5        # 신뢰도
    raw_intensity: float = 0.0     # 원본 강도
    morpheme_modifiers: List[float] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # 감정 출처
    timestamp: int = 0


# ============================================================================
# 양방향 전파 엔진
# ============================================================================

class BidirectionalPropagationEngine:
    """
    양방향 계층 전파 핵심 엔진
    
    수학적 모델:
    
    1. 상향 전파 (Bottom-Up):
       E_up[L] = Σ(E[L-1] × α × w_i)
       where α = up_transmission_ratio (0.7)
    
    2. 하향 전파 (Top-Down):
       E_down[L] = E[L+1] × β^d
       where β = down_decay_base (0.9), d = depth
    
    3. 통합:
       E_final[L] = w_up × E_up[L] + w_down × E_down[L] + w_local × E_local[L]
    """
    
    def __init__(self):
        self.layer_configs = LAYER_CONFIGS
        self.layer_states: Dict[Layer, LayerEmotionState] = {}
        
        # 동적 가중치 (학습 가능!)
        self.dynamic_weights = {
            layer: config.default_weight 
            for layer, config in LAYER_CONFIGS.items()
        }
        
        # 전파 이력
        self.propagation_history = []
    
    def initialize_layer_states(
        self, 
        initial_emotions: Dict[Layer, EmotionVector]
    ):
        """
        각 계층의 초기 감정 상태 설정
        """
        for layer, emotion in initial_emotions.items():
            self.layer_states[layer] = LayerEmotionState(
                layer=layer,
                emotion_vector=emotion.copy(),
                confidence=0.5,
                raw_intensity=emotion.intensity()
            )
    
    # ========================================================================
    # 상향 전파 (Bottom-Up)
    # ========================================================================
    
    def propagate_upward(
        self,
        from_layer: Layer,
        to_layer: Layer
    ) -> EmotionVector:
        """
        하층 → 상층 신호 전달
        
        공식:
            E_transmitted = E_source × α
            where α = up_transmission_ratio
        
        Args:
            from_layer: 출발 계층
            to_layer: 도착 계층
        
        Returns:
            전달된 감정 벡터
        """
        if from_layer not in self.layer_states:
            return EmotionVector()
        
        source_state = self.layer_states[from_layer]
        config = self.layer_configs[from_layer]
        
        # 상향 전달 비율 적용
        transmitted_emotion = self._multiply_emotion_vector(
            source_state.emotion_vector,
            config.up_transmission_ratio
        )
        
        # 신뢰도 반영
        transmitted_emotion = self._multiply_emotion_vector(
            transmitted_emotion,
            source_state.confidence
        )
        
        # 전파 이력 기록
        self.propagation_history.append({
            'type': 'upward',
            'from': from_layer.name,
            'to': to_layer.name,
            'ratio': config.up_transmission_ratio,
            'intensity_before': source_state.raw_intensity,
            'intensity_after': transmitted_emotion.intensity()
        })
        
        return transmitted_emotion
    
    # ========================================================================
    # 하향 전파 (Top-Down)
    # ========================================================================
    
    def propagate_downward(
        self,
        from_layer: Layer,
        to_layer: Layer
    ) -> EmotionVector:
        """
        상층 → 하층 맥락 전달
        
        공식:
            E_transmitted = E_source × β^d
            where β = down_decay_base, d = |level_diff|
        
        핵심:
        - 깊이가 깊을수록 지수적으로 감쇠
        - 담화(5층) → 형태소(1층): 0.9^4 ≈ 0.656
        """
        if from_layer not in self.layer_states:
            return EmotionVector()
        
        source_state = self.layer_states[from_layer]
        config = self.layer_configs[from_layer]
        
        # 계층 간 깊이 계산
        depth = abs(from_layer.value - to_layer.value)
        
        # 지수 감쇠 적용
        decay_factor = config.down_decay_base ** depth
        
        transmitted_emotion = self._multiply_emotion_vector(
            source_state.emotion_vector,
            decay_factor
        )
        
        # 신뢰도 반영
        transmitted_emotion = self._multiply_emotion_vector(
            transmitted_emotion,
            source_state.confidence
        )
        
        # 전파 이력 기록
        self.propagation_history.append({
            'type': 'downward',
            'from': from_layer.name,
            'to': to_layer.name,
            'depth': depth,
            'decay_factor': decay_factor,
            'intensity_before': source_state.raw_intensity,
            'intensity_after': transmitted_emotion.intensity()
        })
        
        return transmitted_emotion
    
    # ========================================================================
    # 양방향 통합
    # ========================================================================
    
    def integrate_bidirectional_signals(
        self,
        layer: Layer,
        local_emotion: EmotionVector,
        upward_emotions: List[EmotionVector],
        downward_emotions: List[EmotionVector]
    ) -> EmotionVector:
        """
        양방향 신호를 통합하여 최종 감정 결정
        
        공식:
            E_final = w_local × E_local 
                    + w_up × mean(E_up_list)
                    + w_down × mean(E_down_list)
        
        가중치 동적 조정:
            - 상향 신호가 강하면 → w_up 증가
            - 하향 신호가 강하면 → w_down 증가
            - 로컬 신뢰도 높으면 → w_local 증가
        """
        # 1. 각 방향 신호 집계
        if upward_emotions:
            avg_upward = self._average_emotion_vectors(upward_emotions)
        else:
            avg_upward = EmotionVector()
        
        if downward_emotions:
            avg_downward = self._average_emotion_vectors(downward_emotions)
        else:
            avg_downward = EmotionVector()
        
        # 2. 신호 강도 측정
        local_intensity = local_emotion.intensity()
        upward_intensity = avg_upward.intensity()
        downward_intensity = avg_downward.intensity()
        
        total_intensity = (local_intensity + upward_intensity + 
                          downward_intensity)
        
        if total_intensity < 1e-6:
            return local_emotion.copy()
        
        # 3. 동적 가중치 계산
        w_local = local_intensity / total_intensity
        w_up = upward_intensity / total_intensity
        w_down = downward_intensity / total_intensity
        
        # 4. 신뢰도 반영
        if layer in self.layer_states:
            confidence = self.layer_states[layer].confidence
            w_local *= (1.0 + confidence * 0.5)  # 신뢰도 높으면 로컬 강조
        
        # 정규화
        total_w = w_local + w_up + w_down
        w_local /= total_w
        w_up /= total_w
        w_down /= total_w
        
        # 5. 통합
        integrated = self._add_emotion_vectors([
            self._multiply_emotion_vector(local_emotion, w_local),
            self._multiply_emotion_vector(avg_upward, w_up),
            self._multiply_emotion_vector(avg_downward, w_down)
        ])
        
        # 6. 중립 제외 정규화
        integrated = integrated.normalize_excluding_neutral()
        
        return integrated
    
    # ========================================================================
    # 전체 전파 실행
    # ========================================================================
    
    def execute_full_propagation(
        self,
        iterations: int = 2
    ) -> Dict[Layer, EmotionVector]:
        """
        양방향 전파를 여러 번 반복 실행
        
        알고리즘:
            for iter in range(iterations):
                # Phase 1: Bottom-Up (형태소 → 담화)
                for layer in [MORPHEME, WORD, PHRASE, SENTENCE]:
                    propagate_upward(layer, layer+1)
                
                # Phase 2: Top-Down (담화 → 형태소)
                for layer in [DISCOURSE, SENTENCE, PHRASE, WORD]:
                    propagate_downward(layer, layer-1)
                
                # Phase 3: Integration
                for all layers:
                    integrate_bidirectional_signals()
        
        Args:
            iterations: 반복 횟수 (기본 2회)
        
        Returns:
            최종 각 계층의 감정 상태
        """
        layers_ordered = [
            Layer.MORPHEME, Layer.WORD, Layer.PHRASE, 
            Layer.SENTENCE, Layer.DISCOURSE
        ]
        
        for iteration in range(iterations):
            print(f"\n{'='*60}")
            print(f"전파 반복 {iteration + 1}/{iterations}")
            print(f"{'='*60}")
            
            # ============================================================
            # Phase 1: 상향 전파 (Bottom-Up)
            # ============================================================
            print("\n[Phase 1] 상향 전파 (Bottom-Up)")
            print("-" * 60)
            
            upward_signals = {layer: [] for layer in layers_ordered}
            
            for i in range(len(layers_ordered) - 1):
                from_layer = layers_ordered[i]
                to_layer = layers_ordered[i + 1]
                
                transmitted = self.propagate_upward(from_layer, to_layer)
                upward_signals[to_layer].append(transmitted)
                
                print(f"{from_layer.name:12} → {to_layer.name:12}: "
                      f"강도 {transmitted.intensity():.3f}")
            
            # ============================================================
            # Phase 2: 하향 전파 (Top-Down)
            # ============================================================
            print("\n[Phase 2] 하향 전파 (Top-Down)")
            print("-" * 60)
            
            downward_signals = {layer: [] for layer in layers_ordered}
            
            for i in range(len(layers_ordered) - 1, 0, -1):
                from_layer = layers_ordered[i]
                to_layer = layers_ordered[i - 1]
                
                transmitted = self.propagate_downward(from_layer, to_layer)
                downward_signals[to_layer].append(transmitted)
                
                depth = abs(from_layer.value - to_layer.value)
                decay = self.layer_configs[from_layer].down_decay_base ** depth
                
                print(f"{from_layer.name:12} → {to_layer.name:12}: "
                      f"강도 {transmitted.intensity():.3f} "
                      f"(감쇠 {decay:.3f})")
            
            # ============================================================
            # Phase 3: 통합
            # ============================================================
            print("\n[Phase 3] 양방향 통합")
            print("-" * 60)
            
            new_states = {}
            
            for layer in layers_ordered:
                if layer not in self.layer_states:
                    continue
                
                current_state = self.layer_states[layer]
                
                integrated_emotion = self.integrate_bidirectional_signals(
                    layer=layer,
                    local_emotion=current_state.emotion_vector,
                    upward_emotions=upward_signals[layer],
                    downward_emotions=downward_signals[layer]
                )
                
                # 상태 업데이트
                new_state = LayerEmotionState(
                    layer=layer,
                    emotion_vector=integrated_emotion,
                    confidence=current_state.confidence,
                    raw_intensity=integrated_emotion.intensity()
                )
                
                new_states[layer] = new_state
                
                print(f"{layer.name:12}: "
                      f"통합 강도 {integrated_emotion.intensity():.3f}, "
                      f"신뢰도 {current_state.confidence:.2f}")
            
            # 상태 업데이트
            self.layer_states.update(new_states)
        
        # 최종 결과 반환
        return {
            layer: state.emotion_vector 
            for layer, state in self.layer_states.items()
        }
    
    # ========================================================================
    # 계층 간 상호작용 분석
    # ========================================================================
    
    def analyze_layer_interactions(self) -> Dict:
        """
        계층 간 상호작용 분석
        
        Returns:
            {
                'dominant_direction': 'upward' or 'downward',
                'interaction_strength': float,
                'layer_contributions': Dict[Layer, float],
                'coherence_score': float
            }
        """
        if not self.propagation_history:
            return {}
        
        # 방향별 신호 강도 집계
        upward_total = sum(
            h['intensity_after'] 
            for h in self.propagation_history 
            if h['type'] == 'upward'
        )
        downward_total = sum(
            h['intensity_after'] 
            for h in self.propagation_history 
            if h['type'] == 'downward'
        )
        
        total = upward_total + downward_total
        if total < 1e-6:
            return {}
        
        # 지배적 방향
        dominant = 'upward' if upward_total > downward_total else 'downward'
        
        # 상호작용 강도
        interaction_strength = min(upward_total, downward_total) / total
        
        # 계층별 기여도
        layer_contributions = {}
        for layer, state in self.layer_states.items():
            layer_contributions[layer.name] = (
                state.raw_intensity * state.confidence
            )
        
        # 일관성 점수 (계층 간 감정 유사도)
        coherence_score = self._calculate_coherence()
        
        return {
            'dominant_direction': dominant,
            'upward_strength': upward_total / total,
            'downward_strength': downward_total / total,
            'interaction_strength': interaction_strength,
            'layer_contributions': layer_contributions,
            'coherence_score': coherence_score
        }
    
    def _calculate_coherence(self) -> float:
        """
        계층 간 감정 일관성 점수
        - 코사인 유사도 평균
        """
        if len(self.layer_states) < 2:
            return 1.0
        
        layers = list(self.layer_states.keys())
        similarities = []
        
        for i in range(len(layers) - 1):
            vec1 = self.layer_states[layers[i]].emotion_vector.to_array()
            vec2 = self.layer_states[layers[i + 1]].emotion_vector.to_array()
            
            # 코사인 유사도
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    # ========================================================================
    # 유틸리티 함수
    # ========================================================================
    
    def _multiply_emotion_vector(
        self, 
        emotion: EmotionVector, 
        scalar: float
    ) -> EmotionVector:
        """감정 벡터에 스칼라 곱"""
        d = emotion.to_dict()
        return EmotionVector.from_dict({
            k: v * scalar for k, v in d.items()
        })
    
    def _add_emotion_vectors(
        self, 
        emotions: List[EmotionVector]
    ) -> EmotionVector:
        """감정 벡터들의 합"""
        if not emotions:
            return EmotionVector()
        
        result_dict = {}
        keys = emotions[0].to_dict().keys()
        
        for key in keys:
            result_dict[key] = sum(e.to_dict()[key] for e in emotions)
        
        return EmotionVector.from_dict(result_dict)
    
    def _average_emotion_vectors(
        self, 
        emotions: List[EmotionVector]
    ) -> EmotionVector:
        """감정 벡터들의 평균"""
        if not emotions:
            return EmotionVector()
        
        summed = self._add_emotion_vectors(emotions)
        return self._multiply_emotion_vector(summed, 1.0 / len(emotions))


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("양방향 계층 전파 시스템 테스트")
    print("="*60)
    
    # 엔진 초기화
    engine = BidirectionalPropagationEngine()
    
    # 초기 감정 설정 (예시)
    initial_emotions = {
        Layer.MORPHEME: EmotionVector(joy=0.3, sadness=0.1),
        Layer.WORD: EmotionVector(joy=0.5, excitement=0.3),
        Layer.PHRASE: EmotionVector(joy=0.4, sadness=0.2),
        Layer.SENTENCE: EmotionVector(sadness=0.6, empathic_pain=0.3),
        Layer.DISCOURSE: EmotionVector(nostalgia=0.5, sadness=0.4)
    }
    
    # 상태 초기화
    engine.initialize_layer_states(initial_emotions)
    
    print("\n초기 상태:")
    print("-" * 60)
    for layer, state in engine.layer_states.items():
        print(f"{layer.name:12}: 강도 {state.raw_intensity:.3f}")
    
    # 양방향 전파 실행
    final_emotions = engine.execute_full_propagation(iterations=2)
    
    # 결과 출력
    print("\n" + "="*60)
    print("최종 결과")
    print("="*60)
    
    for layer, emotion in final_emotions.items():
        top_emotions = sorted(
            emotion.to_dict().items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        print(f"\n{layer.name}:")
        for emo_name, value in top_emotions:
            if value > 0.01:
                print(f"  {emo_name:15}: {value:.3f}")
    
    # 상호작용 분석
    print("\n" + "="*60)
    print("계층 간 상호작용 분석")
    print("="*60)
    
    analysis = engine.analyze_layer_interactions()
    print(f"\n지배적 방향: {analysis['dominant_direction']}")
    print(f"상향 강도: {analysis['upward_strength']:.2%}")
    print(f"하향 강도: {analysis['downward_strength']:.2%}")
    print(f"상호작용 강도: {analysis['interaction_strength']:.3f}")
    print(f"일관성 점수: {analysis['coherence_score']:.3f}")
    
    print("\n계층별 기여도:")
    for layer_name, contribution in analysis['layer_contributions'].items():
        print(f"  {layer_name:12}: {contribution:.3f}")
