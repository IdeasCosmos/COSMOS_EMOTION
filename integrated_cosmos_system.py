"""
COSMOS EMOTION - 최종 통합 시스템
===================================

Complete Integration:
1. 형태소 분석 + 강도 시스템
2. 양방향 계층 전파
3. 5채널 공명 시스템
4. HIT 시스템 (기존)
5. ES Timeline (악보화)

전체 파이프라인:
    텍스트 입력
        ↓
    형태소 분석 (조사/어미 감지)
        ↓
    5개 계층 구축
        ├─ MORPHEME: 형태소 강도
        ├─ WORD: HIT 시스템
        ├─ PHRASE: 구 통합
        ├─ SENTENCE: 문장 분석
        └─ DISCOURSE: 전체 맥락
        ↓
    양방향 전파 (2회 반복)
        ├─ 상향: 0.7배 전달
        └─ 하향: 0.9^depth 감쇠
        ↓
    공명 감지 (5채널)
        ├─ Spectral: 반복 패턴
        ├─ Phase: 타이밍 일치
        ├─ Harmonic: 감정 조화
        ├─ Semantic: 의미 유사
        └─ Cross-Layer: 계층 간
        ↓
    증폭 적용
        ↓
    최종 감정 복합체 + ES Timeline
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import re
from collections import defaultdict

# 이전에 만든 모듈들을 임포트한다고 가정
# from morpheme_intensity_system import *
# from bidirectional_propagation import *
# from resonance_system import *


# ============================================================================
# 통합 데이터 구조
# ============================================================================

@dataclass
class AnalysisResult:
    """전체 분석 결과"""
    text: str
    
    # 계층별 감정
    layer_emotions: Dict[str, 'EmotionVector']
    
    # 공명 패턴
    resonance_patterns: Dict[str, List['ResonancePattern']]
    
    # 증폭 효과
    amplification: Dict[str, float]
    
    # ES Timeline
    timeline: 'ESTimeline'
    
    # 메타데이터
    metadata: Dict
    
    # 디버그 정보
    debug_info: Dict


@dataclass
class ESFrame:
    """Emotion Score 프레임 (악보의 한 박자)"""
    frame_id: int
    timestamp: float          # 초 단위
    
    # 감정 벡터
    emotion_vector: np.ndarray  # 28차원
    
    # 음악 파라미터
    intensity: float          # 0.0 ~ 1.0
    valence: float           # -1.0 ~ 1.0 (긍정/부정)
    arousal: float           # -1.0 ~ 1.0 (흥분/침착)
    tension: float           # 0.0 ~ 1.0 (긴장도)
    
    # 악상
    cadence: str             # 종지 유형
    tempo_bpm: int           # 템포
    dynamic_mark: str        # 악상 기호 (f, p, mf, etc.)
    
    # 공명 정보
    resonance_active: bool
    resonance_channels: List[str]


@dataclass
class ESTimeline:
    """Emotion Score 타임라인 (전체 악보)"""
    fps: int = 25            # Frames Per Second
    frames: List[ESFrame] = field(default_factory=list)
    phrases: List[Dict] = field(default_factory=list)
    
    def duration(self) -> float:
        """총 길이 (초)"""
        return len(self.frames) / self.fps
    
    def add_frame(self, frame: ESFrame):
        """프레임 추가"""
        self.frames.append(frame)
    
    def add_phrase(self, start_frame: int, end_frame: int, label: str):
        """프레이즈 경계 추가"""
        self.phrases.append({
            'id': len(self.phrases),
            'start': start_frame,
            'end': end_frame,
            'label': label,
            'duration': (end_frame - start_frame) / self.fps
        })


# ============================================================================
# 최종 통합 엔진
# ============================================================================

class IntegratedCOSMOSEngine:
    """
    COSMOS EMOTION 완전 통합 엔진
    
    전체 파이프라인을 하나로 통합
    """
    
    def __init__(
        self,
        use_konlpy: bool = True,
        fps: int = 25,
        propagation_iterations: int = 2
    ):
        # 1. 형태소 분석
        from morpheme_intensity_system import (
            MorphemeAnalyzer, 
            MorphemeIntensityEngine
        )
        self.morpheme_analyzer = MorphemeAnalyzer(use_konlpy)
        self.intensity_engine = MorphemeIntensityEngine()
        
        # 2. 양방향 전파
        from bidirectional_propagation import (
            BidirectionalPropagationEngine,
            Layer,
            EmotionVector,
            LayerEmotionState
        )
        self.propagation_engine = BidirectionalPropagationEngine()
        self.Layer = Layer
        self.EmotionVector = EmotionVector
        
        # 3. 공명 시스템
        from resonance_system import (
            MultiChannelResonanceSystem,
            ResonanceSignal
        )
        self.resonance_system = MultiChannelResonanceSystem()
        self.ResonanceSignal = ResonanceSignal
        
        # 4. 설정
        self.fps = fps
        self.propagation_iterations = propagation_iterations
        
        # 5. 감정 사전 (HIT 시스템)
        self.emotion_dictionary = self._load_emotion_dictionary()
    
    def _load_emotion_dictionary(self) -> Dict:
        """
        감정 사전 로드
        - 기존 COSMOS EMOTION의 large_emotion_dictionary.json 사용
        """
        # 간단한 예시 사전 (실제로는 JSON 로드)
        return {
            '기쁘': {'joy': 0.9, 'excitement': 0.3},
            '슬프': {'sadness': 0.9, 'empathic_pain': 0.4},
            '화나': {'anger': 0.9, 'hyeontta': 0.5},
            '짜증': {'anger': 0.7, 'hyeontta': 0.8},
            '좋': {'joy': 0.7, 'excitement': 0.2},
            '싫': {'disgust': 0.7, 'anger': 0.3},
            '무섭': {'fear': 0.9, 'anxiety': 0.6},
            '반가': {'joy': 0.8, 'excitement': 0.5},
            '아프': {'empathic_pain': 0.9, 'sadness': 0.4},
            '외롭': {'sadness': 0.8, 'empathic_pain': 0.5},
        }
    
    # ========================================================================
    # 메인 분석 파이프라인
    # ========================================================================
    
    def analyze(self, text: str) -> AnalysisResult:
        """
        텍스트 → 최종 감정 분석
        
        완전한 파이프라인 실행
        """
        print("="*70)
        print("COSMOS EMOTION - 통합 분석 시작")
        print("="*70)
        print(f"입력: {text}")
        print()
        
        # ====================================================================
        # Phase 1: 형태소 분석
        # ====================================================================
        print("[Phase 1] 형태소 분석")
        print("-" * 70)
        
        morphemes = self.morpheme_analyzer.parse(text)
        
        print(f"형태소 {len(morphemes)}개 감지:")
        for m in morphemes[:10]:  # 처음 10개만
            print(f"  {m.surface} ({m.pos})")
        
        # ====================================================================
        # Phase 2: 5개 계층 구축
        # ====================================================================
        print("\n[Phase 2] 5개 계층 구축")
        print("-" * 70)
        
        layer_emotions = self._build_hierarchical_layers(text, morphemes)
        
        for layer_name, emotion in layer_emotions.items():
            print(f"{layer_name:12}: 강도 {emotion.intensity():.3f}")
        
        # ====================================================================
        # Phase 3: 양방향 전파
        # ====================================================================
        print("\n[Phase 3] 양방향 전파")
        print("-" * 70)
        
        # 엔진 초기화
        self.propagation_engine.initialize_layer_states(layer_emotions)
        
        # 전파 실행
        propagated_emotions = self.propagation_engine.execute_full_propagation(
            iterations=self.propagation_iterations
        )
        
        # ====================================================================
        # Phase 4: 공명 신호 생성
        # ====================================================================
        print("\n[Phase 4] 공명 신호 생성")
        print("-" * 70)
        
        resonance_signals = self._create_resonance_signals(
            propagated_emotions
        )
        
        print(f"공명 신호 {len(resonance_signals)}개 생성")
        
        # ====================================================================
        # Phase 5: 5채널 공명 감지
        # ====================================================================
        print("\n[Phase 5] 5채널 공명 감지")
        print("-" * 70)
        
        resonance_patterns = self.resonance_system.detect_all_resonances(
            resonance_signals
        )
        
        total_patterns = sum(len(p) for p in resonance_patterns.values())
        print(f"\n총 {total_patterns}개 공명 패턴 감지")
        
        # ====================================================================
        # Phase 6: 증폭 적용
        # ====================================================================
        print("\n[Phase 6] 증폭 적용")
        print("-" * 70)
        
        amplification = self.resonance_system.calculate_total_amplification(
            resonance_patterns
        )
        
        print(f"총 증폭률: ×{amplification['total_amplification']:.2f}")
        
        # 최종 감정에 증폭 적용
        final_emotion = self._apply_amplification(
            propagated_emotions[self.Layer.DISCOURSE],
            amplification['total_amplification']
        )
        
        # ====================================================================
        # Phase 7: ES Timeline 생성
        # ====================================================================
        print("\n[Phase 7] ES Timeline 생성 (악보화)")
        print("-" * 70)
        
        timeline = self._generate_timeline(
            text,
            propagated_emotions,
            resonance_patterns,
            amplification
        )
        
        print(f"Timeline: {len(timeline.frames)}프레임, "
              f"{timeline.duration():.2f}초, "
              f"{len(timeline.phrases)}개 프레이즈")
        
        # ====================================================================
        # 결과 통합
        # ====================================================================
        
        # 계층 간 상호작용 분석
        interaction_analysis = \
            self.propagation_engine.analyze_layer_interactions()
        
        result = AnalysisResult(
            text=text,
            layer_emotions=propagated_emotions,
            resonance_patterns=resonance_patterns,
            amplification=amplification,
            timeline=timeline,
            metadata={
                'num_morphemes': len(morphemes),
                'num_resonance_signals': len(resonance_signals),
                'num_resonance_patterns': total_patterns,
                'dominant_direction': interaction_analysis.get('dominant_direction'),
                'coherence_score': interaction_analysis.get('coherence_score', 0.0)
            },
            debug_info={
                'morphemes': morphemes,
                'interaction_analysis': interaction_analysis
            }
        )
        
        return result
    
    # ========================================================================
    # 계층 구축
    # ========================================================================
    
    def _build_hierarchical_layers(
        self,
        text: str,
        morphemes: List
    ) -> Dict:
        """
        5개 계층 감정 구축
        """
        layer_emotions = {}
        
        # ----------------------------------------------------------------
        # Layer 1: MORPHEME (형태소)
        # ----------------------------------------------------------------
        morpheme_emotion = self.EmotionVector()
        
        # 형태소별 강도 계산
        intensity_result = self.intensity_engine.calculate_morpheme_intensity(
            morphemes,
            base_emotion={'joy': 0.5, 'sadness': 0.3}
        )
        
        # 강도 배율을 감정에 반영
        morpheme_emotion.neutral = 0.5 * intensity_result['final_intensity']
        
        layer_emotions[self.Layer.MORPHEME] = morpheme_emotion
        
        # ----------------------------------------------------------------
        # Layer 2: WORD (단어) - HIT 시스템
        # ----------------------------------------------------------------
        word_emotion = self._hit_system_analysis(text)
        layer_emotions[self.Layer.WORD] = word_emotion
        
        # ----------------------------------------------------------------
        # Layer 3: PHRASE (구)
        # ----------------------------------------------------------------
        phrase_emotion = self._phrase_analysis(text, morphemes)
        layer_emotions[self.Layer.PHRASE] = phrase_emotion
        
        # ----------------------------------------------------------------
        # Layer 4: SENTENCE (문장)
        # ----------------------------------------------------------------
        sentence_emotion = self._sentence_analysis(text)
        layer_emotions[self.Layer.SENTENCE] = sentence_emotion
        
        # ----------------------------------------------------------------
        # Layer 5: DISCOURSE (담화)
        # ----------------------------------------------------------------
        discourse_emotion = self._discourse_analysis(text)
        layer_emotions[self.Layer.DISCOURSE] = discourse_emotion
        
        return layer_emotions
    
    def _hit_system_analysis(self, text: str) -> 'EmotionVector':
        """
        HIT 시스템 - 감정 단어 감지
        """
        emotion_dict = self.EmotionVector().to_dict()
        
        # 감정 사전에서 매칭
        for word, emotions in self.emotion_dictionary.items():
            if word in text:
                for emo_name, value in emotions.items():
                    if emo_name in emotion_dict:
                        emotion_dict[emo_name] += value
        
        return self.EmotionVector.from_dict(emotion_dict)
    
    def _phrase_analysis(self, text: str, morphemes: List) -> 'EmotionVector':
        """
        구 단위 분석
        - 여러 단어의 조합
        """
        # 간단히 단어 감정의 평균
        word_emotion = self._hit_system_analysis(text)
        
        # 구 특유의 패턴 감지 (예: "정말 기쁘네요")
        phrase_modifiers = {
            '정말': 1.2,
            '너무': 1.3,
            '진짜': 1.2,
            '완전': 1.4,
        }
        
        modifier = 1.0
        for mod_word, mod_value in phrase_modifiers.items():
            if mod_word in text:
                modifier *= mod_value
        
        emotion_dict = word_emotion.to_dict()
        emotion_dict = {k: v * modifier for k, v in emotion_dict.items()}
        
        return self.EmotionVector.from_dict(emotion_dict)
    
    def _sentence_analysis(self, text: str) -> 'EmotionVector':
        """
        문장 단위 분석
        - 전체적인 감정 흐름
        """
        # 문장 종결 패턴
        if '?' in text:
            return self.EmotionVector(confusion=0.5, curiosity=0.4)
        elif '!' in text:
            return self.EmotionVector(excitement=0.6, joy=0.3)
        else:
            return self.EmotionVector(calmness=0.4)
    
    def _discourse_analysis(self, text: str) -> 'EmotionVector':
        """
        담화 단위 분석
        - 전체 맥락
        """
        # 전체 문맥 파악
        if len(text) > 50:
            # 긴 텍스트 → 복잡한 감정
            return self.EmotionVector(contemplation=0.5, complexity=0.4)
        else:
            # 짧은 텍스트 → 단순 감정
            return self.EmotionVector(neutral=0.5)
    
    # ========================================================================
    # 공명 신호 생성
    # ========================================================================
    
    def _create_resonance_signals(
        self,
        layer_emotions: Dict
    ) -> List:
        """
        계층별 감정 → 공명 신호 변환
        """
        signals = []
        
        for layer, emotion in layer_emotions.items():
            # 위치는 계층 레벨로
            position = layer.value
            
            signal = self.ResonanceSignal(
                position=position,
                layer=layer.name,
                emotion_vector=emotion.to_array(),
                intensity=emotion.intensity(),
                frequency=1.0,  # 기본 주파수
                phase=0.0,      # 기본 위상
                source=f"{layer.name}_layer"
            )
            
            signals.append(signal)
        
        return signals
    
    # ========================================================================
    # 증폭 적용
    # ========================================================================
    
    def _apply_amplification(
        self,
        emotion: 'EmotionVector',
        amplification_factor: float
    ) -> 'EmotionVector':
        """
        공명 증폭을 감정에 적용
        """
        emotion_dict = emotion.to_dict()
        
        # 중립 제외하고 증폭
        neutral_value = emotion_dict.pop('neutral')
        
        amplified = {
            k: v * amplification_factor 
            for k, v in emotion_dict.items()
        }
        
        amplified['neutral'] = neutral_value
        
        return self.EmotionVector.from_dict(amplified)
    
    # ========================================================================
    # ES Timeline 생성
    # ========================================================================
    
    def _generate_timeline(
        self,
        text: str,
        layer_emotions: Dict,
        resonance_patterns: Dict,
        amplification: Dict
    ) -> ESTimeline:
        """
        Emotion Score Timeline 생성
        
        텍스트의 시간적 흐름을 악보로 변환
        """
        timeline = ESTimeline(fps=self.fps)
        
        # 텍스트 길이 기반 프레임 수 결정
        # 1글자당 약 0.2초 (5fps)
        num_frames = max(int(len(text) * 0.2 * self.fps), self.fps)
        
        # 최종 감정 (DISCOURSE 레벨)
        final_emotion = layer_emotions[self.Layer.DISCOURSE]
        
        # 프레임별 생성
        for frame_id in range(num_frames):
            timestamp = frame_id / self.fps
            
            # 감정 벡터 (시간에 따라 변화)
            # 여기서는 단순화: 전체 감정 유지
            emotion_vector = final_emotion.to_array()
            
            # 음악 파라미터 계산
            intensity = final_emotion.intensity()
            valence = self._calculate_valence(final_emotion)
            arousal = self._calculate_arousal(final_emotion)
            tension = self._calculate_tension(frame_id, num_frames)
            
            # 템포 계산 (arousal 기반)
            tempo_bpm = int(60 + arousal * 60)  # 60~120 BPM
            
            # 공명 활성 여부
            resonance_active = any(
                len(patterns) > 0 
                for patterns in resonance_patterns.values()
            )
            
            frame = ESFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                emotion_vector=emotion_vector,
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                tension=tension,
                cadence=self._determine_cadence(frame_id, num_frames),
                tempo_bpm=tempo_bpm,
                dynamic_mark=self._determine_dynamic_mark(intensity),
                resonance_active=resonance_active,
                resonance_channels=list(resonance_patterns.keys())
            )
            
            timeline.add_frame(frame)
        
        # 프레이즈 경계 추가
        # 문장 부호 기준
        timeline.add_phrase(0, num_frames, text)
        
        return timeline
    
    def _calculate_valence(self, emotion: 'EmotionVector') -> float:
        """
        Valence 계산 (긍정/부정)
        
        -1.0 (매우 부정) ~ +1.0 (매우 긍정)
        """
        positive = emotion.joy + emotion.excitement + emotion.relief
        negative = emotion.sadness + emotion.anger + emotion.fear
        
        total = positive + negative
        if total < 1e-6:
            return 0.0
        
        return (positive - negative) / total
    
    def _calculate_arousal(self, emotion: 'EmotionVector') -> float:
        """
        Arousal 계산 (흥분/침착)
        
        -1.0 (매우 침착) ~ +1.0 (매우 흥분)
        """
        excited = emotion.excitement + emotion.anger + emotion.fear
        calm = emotion.calmness + emotion.relief + emotion.neutral
        
        total = excited + calm
        if total < 1e-6:
            return 0.0
        
        return (excited - calm) / total
    
    def _calculate_tension(self, frame_id: int, total_frames: int) -> float:
        """
        Tension 계산 (긴장도)
        
        시간에 따라 변화하는 긴장감
        """
        # 중간에 최고조, 시작/끝에 낮음
        progress = frame_id / total_frames
        
        # 포물선 형태
        tension = 4 * progress * (1 - progress)
        
        return float(tension)
    
    def _determine_cadence(self, frame_id: int, total_frames: int) -> str:
        """
        Cadence(종지) 결정
        """
        progress = frame_id / total_frames
        
        if progress < 0.3:
            return "intro"
        elif progress < 0.7:
            return "development"
        elif progress < 0.9:
            return "climax"
        else:
            return "resolution"
    
    def _determine_dynamic_mark(self, intensity: float) -> str:
        """
        Dynamic Mark(악상 기호) 결정
        """
        if intensity < 0.2:
            return "pp (매우 약하게)"
        elif intensity < 0.4:
            return "p (약하게)"
        elif intensity < 0.6:
            return "mf (조금 세게)"
        elif intensity < 0.8:
            return "f (세게)"
        else:
            return "ff (매우 세게)"
    
    # ========================================================================
    # 결과 출력
    # ========================================================================
    
    def print_result(self, result: AnalysisResult):
        """
        분석 결과를 보기 좋게 출력
        """
        print("\n" + "="*70)
        print("COSMOS EMOTION - 최종 분석 결과")
        print("="*70)
        
        print(f"\n입력: {result.text}")
        
        # 계층별 감정
        print("\n[계층별 감정]")
        print("-" * 70)
        for layer, emotion in result.layer_emotions.items():
            top_emotions = sorted(
                emotion.to_dict().items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            print(f"\n{layer.name:12}:")
            for emo_name, value in top_emotions:
                if value > 0.01:
                    bar = "█" * int(value * 20)
                    print(f"  {emo_name:15} {value:5.2f} {bar}")
        
        # 공명 패턴
        print("\n[공명 패턴]")
        print("-" * 70)
        total_patterns = sum(
            len(p) for p in result.resonance_patterns.values()
        )
        print(f"총 {total_patterns}개 감지")
        
        for channel, patterns in result.resonance_patterns.items():
            if patterns:
                print(f"\n{channel}:")
                for i, pattern in enumerate(patterns, 1):
                    print(f"  패턴 {i}: "
                          f"강도 {pattern.resonance_strength:.2f}, "
                          f"증폭 ×{pattern.amplification:.2f}")
        
        # 증폭 효과
        print("\n[증폭 효과]")
        print("-" * 70)
        print(f"총 증폭률: ×{result.amplification['total_amplification']:.2f}")
        
        print("\n채널별 기여:")
        for channel, contribution in \
                result.amplification['channel_contributions'].items():
            print(f"  {channel:15}: +{contribution*100:.1f}%")
        
        # Timeline 정보
        print("\n[ES Timeline]")
        print("-" * 70)
        print(f"총 길이: {result.timeline.duration():.2f}초")
        print(f"프레임: {len(result.timeline.frames)}개")
        print(f"프레이즈: {len(result.timeline.phrases)}개")
        
        # 메타데이터
        print("\n[메타데이터]")
        print("-" * 70)
        for key, value in result.metadata.items():
            print(f"{key:25}: {value}")


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    # 엔진 초기화
    engine = IntegratedCOSMOSEngine(
        use_konlpy=False,  # 자체 파서 사용
        fps=25,
        propagation_iterations=2
    )
    
    # 테스트 문장들
    test_sentences = [
        "정말 기쁘네요! 오늘은 최고의 날이에요.",
        "슬프지만 견뎌야 해.",
        "너무 짜증나ㅋㅋㅋ",
        "오래된 앨범 속에서 환하게 웃고 있는 친구의 모습을 보니 반가웠지만, "
        "이제는 다시 볼 수 없다는 생각에 가슴 한편이 아려왔다."
    ]
    
    # 각 문장 분석
    for i, sentence in enumerate(test_sentences, 1):
        print("\n\n" + "="*70)
        print(f"테스트 {i}/{len(test_sentences)}")
        print("="*70)
        
        result = engine.analyze(sentence)
        engine.print_result(result)
        
        print("\n" + "="*70)
        print("분석 완료!")
        print("="*70)
