"""
COSMOS EMOTION - Resonance 공명 시스템
======================================

Part 3: Multi-Channel Resonance Detection System
- 5채널 공명 감지: Spectral, Phase, Harmonic, Semantic, Cross-Layer
- 신경망 확장 대비 설계
- 증폭/감쇠 효과 모델링
- 공명 간섭 패턴 분석

🎵 음악적 비유:

공명(Resonance)이란?
- 같은 주파수의 소리가 만나면 → 증폭!
- 반대 위상이 만나면 → 상쇄!
- 화음이 맞으면 → 아름다운 울림!

감정도 마찬가지:
- 여러 층에서 같은 감정 → 공명 증폭
- 반대 감정 충돌 → 복합 감정 생성
- 타이밍 일치 → 강렬한 임팩트
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from scipy import signal
from scipy.spatial.distance import cosine
import math


# ============================================================================
# 공명 채널 정의
# ============================================================================

class ResonanceChannel:
    """공명 채널 열거형"""
    SPECTRAL = "spectral"       # 주파수 공명 (같은 감정 반복)
    PHASE = "phase"             # 위상 공명 (타이밍 일치)
    HARMONIC = "harmonic"       # 화음 공명 (감정 조화)
    SEMANTIC = "semantic"       # 의미 공명 (문맥 유사도)
    CROSS_LAYER = "cross_layer" # 계층 간 공명 (수직 연결)


@dataclass
class ResonanceSignal:
    """
    공명 신호 정보
    
    신경망 확장을 위한 Feature Vector 포함:
    - 28차원 감정 벡터
    - 공명 강도
    - 주파수/위상 정보
    - 문맥 임베딩
    """
    position: int              # 위치 (시간축)
    layer: str                 # 계층
    emotion_vector: np.ndarray # 28차원 감정 벡터
    intensity: float           # 강도
    frequency: float           # 주파수 (반복 패턴)
    phase: float              # 위상 (0 ~ 2π)
    context_embedding: Optional[np.ndarray] = None  # 문맥 임베딩 (신경망용)
    
    # 메타데이터
    source: str = ""
    timestamp: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        """
        신경망 학습용 Feature Vector 생성
        
        구조:
        [emotion_vector(28), intensity(1), frequency(1), phase(1), 
         context_embedding(128)] = 총 159차원
        """
        features = [
            self.emotion_vector,                    # 28차원
            np.array([self.intensity]),             # 1차원
            np.array([self.frequency]),             # 1차원
            np.array([self.phase]),                 # 1차원
        ]
        
        if self.context_embedding is not None:
            features.append(self.context_embedding)  # 128차원
        else:
            features.append(np.zeros(128))           # 빈 임베딩
        
        return np.concatenate(features)


@dataclass
class ResonancePattern:
    """
    감지된 공명 패턴
    """
    channel: str               # 공명 채널
    signals: List[ResonanceSignal]  # 참여 신호들
    resonance_strength: float  # 공명 강도 (0.0 ~ 2.0+)
    amplification: float       # 증폭률 (1.0 = 변화 없음)
    pattern_type: str          # 패턴 유형
    confidence: float          # 신뢰도
    
    # 신경망 학습용
    feature_matrix: Optional[np.ndarray] = None


# ============================================================================
# 1. Spectral Resonance (주파수 공명)
# ============================================================================

class SpectralResonanceDetector:
    """
    주파수 공명 감지기
    
    원리:
    - 같은 감정이 반복되면 → 주파수 공명 발생
    - 공명 강도 ∝ 반복 횟수 × 일관성
    
    신경망 확장:
    - LSTM으로 시퀀스 패턴 학습
    - Attention으로 중요 반복 강조
    """
    
    def __init__(self, min_repetitions: int = 3, window_size: int = 10):
        self.min_repetitions = min_repetitions
        self.window_size = window_size
    
    def detect(
        self, 
        signals: List[ResonanceSignal]
    ) -> List[ResonancePattern]:
        """
        주파수 공명 감지
        
        알고리즘:
        1. 슬라이딩 윈도우로 신호 분석
        2. 같은 감정의 반복 횟수 계산
        3. 반복 패턴의 규칙성 평가
        4. 공명 강도 계산
        """
        patterns = []
        
        # 감정 유형별 신호 그룹화
        emotion_groups = defaultdict(list)
        
        for sig in signals:
            # 주 감정 추출 (가장 강한 감정)
            dominant_emotion = self._get_dominant_emotion(sig.emotion_vector)
            emotion_groups[dominant_emotion].append(sig)
        
        # 각 감정 그룹에 대해 공명 검사
        for emotion_type, emotion_signals in emotion_groups.items():
            if len(emotion_signals) < self.min_repetitions:
                continue
            
            # 반복 패턴 분석
            repetition_pattern = self._analyze_repetition_pattern(
                emotion_signals
            )
            
            if repetition_pattern['is_resonating']:
                # 공명 강도 계산
                resonance_strength = self._calculate_spectral_strength(
                    repetition_pattern
                )
                
                # 증폭률 계산
                amplification = 1.0 + (len(emotion_signals) - 2) * 0.15
                
                pattern = ResonancePattern(
                    channel=ResonanceChannel.SPECTRAL,
                    signals=emotion_signals,
                    resonance_strength=resonance_strength,
                    amplification=amplification,
                    pattern_type=f"repetition_{emotion_type}",
                    confidence=repetition_pattern['confidence'],
                    feature_matrix=self._create_feature_matrix(emotion_signals)
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _get_dominant_emotion(self, emotion_vector: np.ndarray) -> str:
        """주 감정 추출"""
        emotion_names = [
            'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise',
            'neutral', 'han', 'jeong', 'nunchi', 'hyeontta', 'menboong',
            'excitement', 'calmness', 'empathic_pain', 'amusement',
            'confusion', 'disappointment', 'guilt', 'shame', 'pride',
            'relief', 'hope', 'despair', 'nostalgia', 'contempt',
            'envy', 'gratitude'
        ]
        
        max_idx = np.argmax(emotion_vector)
        return emotion_names[max_idx] if max_idx < len(emotion_names) else 'unknown'
    
    def _analyze_repetition_pattern(
        self, 
        signals: List[ResonanceSignal]
    ) -> Dict:
        """
        반복 패턴 분석
        
        Returns:
            {
                'is_resonating': bool,
                'regularity': float,  # 규칙성 (0~1)
                'confidence': float,
                'period': float       # 주기
            }
        """
        if len(signals) < 2:
            return {'is_resonating': False, 'confidence': 0.0}
        
        # 위치 간격 계산
        positions = [sig.position for sig in signals]
        intervals = [positions[i+1] - positions[i] 
                    for i in range(len(positions)-1)]
        
        if not intervals:
            return {'is_resonating': False, 'confidence': 0.0}
        
        # 간격의 규칙성 평가 (표준편차가 작을수록 규칙적)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        regularity = 1.0 / (1.0 + std_interval / (mean_interval + 1e-6))
        
        # 공명 판정
        is_resonating = (
            len(signals) >= self.min_repetitions and
            regularity > 0.5
        )
        
        return {
            'is_resonating': is_resonating,
            'regularity': regularity,
            'confidence': regularity,
            'period': mean_interval
        }
    
    def _calculate_spectral_strength(
        self, 
        pattern: Dict
    ) -> float:
        """
        주파수 공명 강도 계산
        
        공식:
            S = regularity × log(1 + repetitions)
        """
        regularity = pattern['regularity']
        # repetitions는 패턴에서 추론
        repetitions = pattern.get('repetitions', 3)
        
        strength = regularity * np.log1p(repetitions)
        return float(strength)
    
    def _create_feature_matrix(
        self, 
        signals: List[ResonanceSignal]
    ) -> np.ndarray:
        """
        신경망 학습용 Feature Matrix 생성
        
        Shape: (n_signals, feature_dim)
        """
        return np.array([sig.to_feature_vector() for sig in signals])


# ============================================================================
# 2. Phase Resonance (위상 공명)
# ============================================================================

class PhaseResonanceDetector:
    """
    위상 공명 감지기
    
    원리:
    - 감정 변화 타이밍이 일치하면 → 위상 공명
    - 여러 감정이 동시에 변화 → 강한 임팩트
    
    신경망 확장:
    - Transformer의 Temporal Attention
    - 시간 축 상관관계 학습
    """
    
    def __init__(self, sync_threshold: float = 2.0):
        self.sync_threshold = sync_threshold  # 동시 판정 기준 (위치 차이)
    
    def detect(
        self, 
        signals: List[ResonanceSignal]
    ) -> List[ResonancePattern]:
        """
        위상 공명 감지
        
        알고리즘:
        1. 신호들을 시간축으로 정렬
        2. 근접한 신호들을 클러스터링
        3. 각 클러스터의 위상 동기화 정도 평가
        4. 공명 강도 계산
        """
        patterns = []
        
        if len(signals) < 2:
            return patterns
        
        # 위치별 클러스터링
        clusters = self._cluster_by_position(signals)
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # 위상 동기화 분석
            sync_analysis = self._analyze_phase_synchronization(cluster)
            
            if sync_analysis['is_synchronized']:
                # 공명 강도 계산
                resonance_strength = self._calculate_phase_strength(
                    sync_analysis
                )
                
                # 증폭률 (동시 발생 신호 수에 비례)
                amplification = 1.0 + len(cluster) * 0.2
                
                pattern = ResonancePattern(
                    channel=ResonanceChannel.PHASE,
                    signals=cluster,
                    resonance_strength=resonance_strength,
                    amplification=amplification,
                    pattern_type="phase_sync",
                    confidence=sync_analysis['confidence'],
                    feature_matrix=self._create_feature_matrix(cluster)
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _cluster_by_position(
        self, 
        signals: List[ResonanceSignal]
    ) -> List[List[ResonanceSignal]]:
        """
        위치 기반 클러스터링
        
        DBSCAN과 유사한 접근
        """
        sorted_signals = sorted(signals, key=lambda s: s.position)
        
        clusters = []
        current_cluster = [sorted_signals[0]]
        
        for i in range(1, len(sorted_signals)):
            pos_diff = sorted_signals[i].position - sorted_signals[i-1].position
            
            if pos_diff <= self.sync_threshold:
                current_cluster.append(sorted_signals[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [sorted_signals[i]]
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        return clusters
    
    def _analyze_phase_synchronization(
        self, 
        cluster: List[ResonanceSignal]
    ) -> Dict:
        """
        위상 동기화 분석
        
        Returns:
            {
                'is_synchronized': bool,
                'sync_strength': float,
                'confidence': float,
                'mean_phase': float,
                'phase_variance': float
            }
        """
        phases = [sig.phase for sig in cluster]
        
        # 위상 평균 (circular mean)
        mean_phase = np.arctan2(
            np.mean(np.sin(phases)),
            np.mean(np.cos(phases))
        )
        
        # 위상 분산 (circular variance)
        phase_variance = 1 - np.sqrt(
            np.mean(np.sin(phases))**2 + np.mean(np.cos(phases))**2
        )
        
        # 동기화 강도 (분산이 작을수록 강함)
        sync_strength = 1.0 - phase_variance
        
        is_synchronized = sync_strength > 0.7
        
        return {
            'is_synchronized': is_synchronized,
            'sync_strength': sync_strength,
            'confidence': sync_strength,
            'mean_phase': mean_phase,
            'phase_variance': phase_variance
        }
    
    def _calculate_phase_strength(self, analysis: Dict) -> float:
        """위상 공명 강도 계산"""
        return analysis['sync_strength']
    
    def _create_feature_matrix(
        self, 
        signals: List[ResonanceSignal]
    ) -> np.ndarray:
        """Feature Matrix 생성"""
        return np.array([sig.to_feature_vector() for sig in signals])


# ============================================================================
# 3. Harmonic Resonance (화음 공명)
# ============================================================================

class HarmonicResonanceDetector:
    """
    화음 공명 감지기
    
    원리:
    - 조화로운 감정 조합 → 화음 공명
    - 예: joy + excitement, sadness + empathic_pain
    - 불협화 감지 → 복합 감정
    
    신경망 확장:
    - Graph Neural Network로 감정 간 관계 학습
    - Emotion Harmony Matrix 학습
    """
    
    def __init__(self):
        # 감정 조화 매트릭스 (사전 정의 + 학습 가능)
        self.harmony_matrix = self._initialize_harmony_matrix()
    
    def _initialize_harmony_matrix(self) -> np.ndarray:
        """
        감정 조화 매트릭스 초기화
        
        28×28 행렬: harmony_matrix[i][j] = i와 j의 조화도
        - 1.0: 완전 조화 (같은 계열)
        - 0.5: 중립
        - 0.0: 불협화 (반대 감정)
        
        신경망으로 학습 가능!
        """
        n_emotions = 28
        matrix = np.eye(n_emotions) * 1.0  # 대각선 1.0
        
        # 조화로운 감정 쌍 (예시)
        harmonious_pairs = [
            (0, 12),   # joy - excitement
            (1, 14),   # sadness - empathic_pain
            (2, 10),   # anger - hyeontta
            (3, 16),   # fear - confusion
            (7, 8),    # han - jeong
            (22, 24),  # hope - relief
        ]
        
        for i, j in harmonious_pairs:
            matrix[i, j] = 0.9
            matrix[j, i] = 0.9
        
        # 불협화 감정 쌍
        dissonant_pairs = [
            (0, 1),    # joy - sadness
            (0, 2),    # joy - anger
            (12, 13),  # excitement - calmness
            (22, 23),  # hope - despair
        ]
        
        for i, j in dissonant_pairs:
            matrix[i, j] = 0.1
            matrix[j, i] = 0.1
        
        return matrix
    
    def detect(
        self, 
        signals: List[ResonanceSignal]
    ) -> List[ResonancePattern]:
        """
        화음 공명 감지
        """
        patterns = []
        
        if len(signals) < 2:
            return patterns
        
        # 신호 쌍별 조화도 계산
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                harmony_analysis = self._analyze_harmony(
                    signals[i], signals[j]
                )
                
                if harmony_analysis['is_harmonic']:
                    pattern = ResonancePattern(
                        channel=ResonanceChannel.HARMONIC,
                        signals=[signals[i], signals[j]],
                        resonance_strength=harmony_analysis['harmony_score'],
                        amplification=harmony_analysis['amplification'],
                        pattern_type=harmony_analysis['type'],
                        confidence=harmony_analysis['confidence'],
                        feature_matrix=self._create_feature_matrix(
                            [signals[i], signals[j]]
                        )
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_harmony(
        self, 
        sig1: ResonanceSignal, 
        sig2: ResonanceSignal
    ) -> Dict:
        """
        두 신호의 조화도 분석
        """
        vec1 = sig1.emotion_vector
        vec2 = sig2.emotion_vector
        
        # 주 감정 인덱스
        idx1 = np.argmax(vec1)
        idx2 = np.argmax(vec2)
        
        # 조화 매트릭스에서 조회
        harmony_score = self.harmony_matrix[idx1, idx2]
        
        # 강도 고려
        combined_intensity = sig1.intensity * sig2.intensity
        
        # 조화 판정
        is_harmonic = harmony_score > 0.7
        is_dissonant = harmony_score < 0.3
        
        if is_harmonic:
            pattern_type = "consonance"
            amplification = 1.0 + harmony_score * 0.5
        elif is_dissonant:
            pattern_type = "dissonance"
            amplification = 1.0  # 불협화는 증폭 없음, 복합감정 생성
        else:
            pattern_type = "neutral"
            amplification = 1.0
        
        return {
            'is_harmonic': is_harmonic or is_dissonant,
            'harmony_score': harmony_score,
            'amplification': amplification,
            'type': pattern_type,
            'confidence': abs(harmony_score - 0.5) * 2  # 극단적일수록 확신
        }
    
    def _create_feature_matrix(
        self, 
        signals: List[ResonanceSignal]
    ) -> np.ndarray:
        """Feature Matrix 생성"""
        return np.array([sig.to_feature_vector() for sig in signals])


# ============================================================================
# 4. Semantic Resonance (의미 공명)
# ============================================================================

class SemanticResonanceDetector:
    """
    의미 공명 감지기
    
    원리:
    - 문맥적으로 유사한 감정들 → 의미 공명
    - 코사인 유사도 기반
    
    신경망 확장:
    - BERT/GPT 임베딩 활용
    - Sentence Transformers
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
    
    def detect(
        self, 
        signals: List[ResonanceSignal]
    ) -> List[ResonancePattern]:
        """
        의미 공명 감지
        """
        patterns = []
        
        if len(signals) < 2:
            return patterns
        
        # 모든 신호 쌍에 대해 유사도 계산
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                similarity = self._calculate_semantic_similarity(
                    signals[i], signals[j]
                )
                
                if similarity > self.similarity_threshold:
                    pattern = ResonancePattern(
                        channel=ResonanceChannel.SEMANTIC,
                        signals=[signals[i], signals[j]],
                        resonance_strength=similarity,
                        amplification=1.0 + similarity * 0.3,
                        pattern_type="semantic_match",
                        confidence=similarity,
                        feature_matrix=self._create_feature_matrix(
                            [signals[i], signals[j]]
                        )
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_semantic_similarity(
        self, 
        sig1: ResonanceSignal, 
        sig2: ResonanceSignal
    ) -> float:
        """
        의미적 유사도 계산
        
        방법:
        1. Context Embedding이 있으면 → 코사인 유사도
        2. 없으면 → 감정 벡터 유사도
        """
        # Context Embedding 사용
        if (sig1.context_embedding is not None and 
            sig2.context_embedding is not None):
            
            similarity = 1 - cosine(
                sig1.context_embedding, 
                sig2.context_embedding
            )
        else:
            # 감정 벡터 사용
            similarity = 1 - cosine(
                sig1.emotion_vector, 
                sig2.emotion_vector
            )
        
        return float(similarity)
    
    def _create_feature_matrix(
        self, 
        signals: List[ResonanceSignal]
    ) -> np.ndarray:
        """Feature Matrix 생성"""
        return np.array([sig.to_feature_vector() for sig in signals])


# ============================================================================
# 5. Cross-Layer Resonance (계층 간 공명)
# ============================================================================

class CrossLayerResonanceDetector:
    """
    계층 간 공명 감지기
    
    원리:
    - 여러 계층에서 같은 감정 → 수직 공명
    - 강력한 증폭 효과
    
    신경망 확장:
    - Hierarchical Attention Network
    - Multi-Scale Feature Fusion
    """
    
    def __init__(self, min_layers: int = 3):
        self.min_layers = min_layers
    
    def detect(
        self, 
        signals: List[ResonanceSignal]
    ) -> List[ResonancePattern]:
        """
        계층 간 공명 감지
        """
        patterns = []
        
        # 계층별로 신호 그룹화
        layer_groups = defaultdict(list)
        for sig in signals:
            layer_groups[sig.layer].append(sig)
        
        if len(layer_groups) < self.min_layers:
            return patterns
        
        # 각 감정 유형별로 계층 간 일치 검사
        emotion_layer_map = defaultdict(lambda: defaultdict(list))
        
        for layer, layer_signals in layer_groups.items():
            for sig in layer_signals:
                dominant = self._get_dominant_emotion(sig.emotion_vector)
                emotion_layer_map[dominant][layer].append(sig)
        
        # 다층 감정 검사
        for emotion_type, layers_dict in emotion_layer_map.items():
            if len(layers_dict) >= self.min_layers:
                # 공명 발생!
                all_signals = []
                for layer_signals in layers_dict.values():
                    all_signals.extend(layer_signals)
                
                resonance_strength = len(layers_dict) / 5.0  # 최대 5층
                amplification = 1.0 + (len(layers_dict) - 2) * 0.3
                
                pattern = ResonancePattern(
                    channel=ResonanceChannel.CROSS_LAYER,
                    signals=all_signals,
                    resonance_strength=resonance_strength,
                    amplification=amplification,
                    pattern_type=f"multi_layer_{emotion_type}",
                    confidence=resonance_strength,
                    feature_matrix=self._create_feature_matrix(all_signals)
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _get_dominant_emotion(self, emotion_vector: np.ndarray) -> str:
        """주 감정 추출"""
        emotion_names = [
            'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise',
            'neutral', 'han', 'jeong', 'nunchi', 'hyeontta', 'menboong',
            'excitement', 'calmness', 'empathic_pain', 'amusement',
            'confusion', 'disappointment', 'guilt', 'shame', 'pride',
            'relief', 'hope', 'despair', 'nostalgia', 'contempt',
            'envy', 'gratitude'
        ]
        
        max_idx = np.argmax(emotion_vector)
        return emotion_names[max_idx] if max_idx < len(emotion_names) else 'unknown'
    
    def _create_feature_matrix(
        self, 
        signals: List[ResonanceSignal]
    ) -> np.ndarray:
        """Feature Matrix 생성"""
        return np.array([sig.to_feature_vector() for sig in signals])


# ============================================================================
# 통합 공명 시스템
# ============================================================================

class MultiChannelResonanceSystem:
    """
    5채널 공명 시스템 통합
    
    신경망 확장 로드맵:
    1. Phase 1: 규칙 기반 (현재)
    2. Phase 2: 하이브리드 (규칙 + 학습된 가중치)
    3. Phase 3: End-to-End 학습 (Transformer + GNN)
    """
    
    def __init__(self):
        self.spectral_detector = SpectralResonanceDetector()
        self.phase_detector = PhaseResonanceDetector()
        self.harmonic_detector = HarmonicResonanceDetector()
        self.semantic_detector = SemanticResonanceDetector()
        self.cross_layer_detector = CrossLayerResonanceDetector()
        
        # 채널별 가중치 (학습 가능!)
        self.channel_weights = {
            ResonanceChannel.SPECTRAL: 1.0,
            ResonanceChannel.PHASE: 1.2,
            ResonanceChannel.HARMONIC: 1.5,
            ResonanceChannel.SEMANTIC: 1.1,
            ResonanceChannel.CROSS_LAYER: 1.8  # 가장 중요!
        }
    
    def detect_all_resonances(
        self, 
        signals: List[ResonanceSignal]
    ) -> Dict[str, List[ResonancePattern]]:
        """
        모든 채널에서 공명 감지
        """
        results = {}
        
        print("\n" + "="*60)
        print("5채널 공명 감지 시작")
        print("="*60)
        
        # 1. Spectral
        print("\n[1/5] Spectral Resonance (주파수 공명)...")
        results[ResonanceChannel.SPECTRAL] = \
            self.spectral_detector.detect(signals)
        print(f"  감지: {len(results[ResonanceChannel.SPECTRAL])}개")
        
        # 2. Phase
        print("\n[2/5] Phase Resonance (위상 공명)...")
        results[ResonanceChannel.PHASE] = \
            self.phase_detector.detect(signals)
        print(f"  감지: {len(results[ResonanceChannel.PHASE])}개")
        
        # 3. Harmonic
        print("\n[3/5] Harmonic Resonance (화음 공명)...")
        results[ResonanceChannel.HARMONIC] = \
            self.harmonic_detector.detect(signals)
        print(f"  감지: {len(results[ResonanceChannel.HARMONIC])}개")
        
        # 4. Semantic
        print("\n[4/5] Semantic Resonance (의미 공명)...")
        results[ResonanceChannel.SEMANTIC] = \
            self.semantic_detector.detect(signals)
        print(f"  감지: {len(results[ResonanceChannel.SEMANTIC])}개")
        
        # 5. Cross-Layer
        print("\n[5/5] Cross-Layer Resonance (계층 간 공명)...")
        results[ResonanceChannel.CROSS_LAYER] = \
            self.cross_layer_detector.detect(signals)
        print(f"  감지: {len(results[ResonanceChannel.CROSS_LAYER])}개")
        
        return results
    
    def calculate_total_amplification(
        self, 
        resonance_results: Dict[str, List[ResonancePattern]]
    ) -> Dict[str, float]:
        """
        총 증폭률 계산
        
        공식:
            A_total = Π(1 + w_i × a_i)
            where w_i = 채널 가중치, a_i = 증폭률
        """
        total_amp = 1.0
        channel_contributions = {}
        
        for channel, patterns in resonance_results.items():
            if not patterns:
                channel_contributions[channel] = 0.0
                continue
            
            # 해당 채널의 평균 증폭률
            avg_amp = np.mean([p.amplification for p in patterns])
            
            # 가중치 적용
            weighted_amp = self.channel_weights[channel] * (avg_amp - 1.0)
            
            total_amp *= (1.0 + weighted_amp)
            channel_contributions[channel] = weighted_amp
        
        return {
            'total_amplification': total_amp,
            'channel_contributions': channel_contributions
        }
    
    def apply_resonance_to_emotion(
        self,
        base_emotion: np.ndarray,
        resonance_results: Dict[str, List[ResonancePattern]]
    ) -> np.ndarray:
        """
        공명 효과를 감정에 적용
        """
        amplification = self.calculate_total_amplification(resonance_results)
        
        # 증폭 적용
        amplified_emotion = base_emotion * amplification['total_amplification']
        
        # 정규화 (중립 제외)
        # ... (생략)
        
        return amplified_emotion


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Multi-Channel Resonance System 테스트")
    print("="*60)
    
    # 테스트 신호 생성
    test_signals = [
        ResonanceSignal(
            position=0,
            layer="WORD",
            emotion_vector=np.array([0.8, 0.1, 0, 0, 0, 0, 0.1] + [0]*21),
            intensity=0.8,
            frequency=1.0,
            phase=0.0,
            source="기쁘다"
        ),
        ResonanceSignal(
            position=5,
            layer="PHRASE",
            emotion_vector=np.array([0.7, 0.2, 0, 0, 0, 0, 0.1] + [0]*21),
            intensity=0.7,
            frequency=1.0,
            phase=0.1,
            source="정말 기쁘네요"
        ),
        ResonanceSignal(
            position=8,
            layer="SENTENCE",
            emotion_vector=np.array([0.9, 0.05, 0, 0, 0, 0, 0.05] + [0]*21),
            intensity=0.9,
            frequency=1.0,
            phase=0.0,
            source="오늘은 최고의 날"
        ),
    ]
    
    # 공명 시스템 초기화
    resonance_system = MultiChannelResonanceSystem()
    
    # 공명 감지
    results = resonance_system.detect_all_resonances(test_signals)
    
    # 결과 출력
    print("\n" + "="*60)
    print("공명 감지 결과")
    print("="*60)
    
    total_patterns = sum(len(patterns) for patterns in results.values())
    print(f"\n총 {total_patterns}개의 공명 패턴 감지")
    
    for channel, patterns in results.items():
        if patterns:
            print(f"\n{channel}:")
            for i, pattern in enumerate(patterns, 1):
                print(f"  패턴 {i}:")
                print(f"    강도: {pattern.resonance_strength:.3f}")
                print(f"    증폭: ×{pattern.amplification:.2f}")
                print(f"    신뢰도: {pattern.confidence:.2%}")
    
    # 총 증폭률 계산
    amplification = resonance_system.calculate_total_amplification(results)
    
    print("\n" + "="*60)
    print("총 증폭 효과")
    print("="*60)
    print(f"총 증폭률: ×{amplification['total_amplification']:.2f}")
    print("\n채널별 기여도:")
    for channel, contribution in amplification['channel_contributions'].items():
        print(f"  {channel:15}: +{contribution:.2%}")
