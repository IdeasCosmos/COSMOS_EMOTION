"""
COSMOS EMOTION - 형태소 강도 분석 시스템
===========================================

Part 1: 형태소 분석기 + 조사/어미 강도 시스템
- 최적 구현 방안: KoNLPy + 자체 규칙 엔진 하이브리드
- 조사/어미별 감정 강도/극성/전환 규칙
- 인터넷 표현 감지 및 강도 매핑
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# ============================================================================
# 형태소 분석 전략
# ============================================================================
"""
최적 구현 방안:

Option 1: KoNLPy (Okt/Komoran) ✅ 추천
  - 장점: 정확도 높음, 품사 태깅 완벽
  - 단점: 설치 필요 (Java 의존성)
  
Option 2: 자체 규칙 기반 파서
  - 장점: 의존성 없음, 빠름
  - 단점: 복잡한 문장 처리 제한
  
Option 3: 하이브리드 (채택!) ⭐
  - KoNLPy 있으면 사용, 없으면 자체 파서
  - 최고의 호환성과 성능
"""


@dataclass
class Morpheme:
    """형태소 정보 구조체"""
    surface: str        # 표면형 (예: "했어요")
    lemma: str         # 기본형 (예: "하다")
    pos: str           # 품사 (예: "VV", "JX", "EF")
    position: int      # 문장 내 위치
    original_index: Tuple[int, int]  # 원문 시작/끝 인덱스
    

class MorphemeAnalyzer:
    """
    형태소 분석기
    - KoNLPy 우선 사용, 없으면 자체 파서
    """
    
    def __init__(self, use_konlpy: bool = True):
        self.use_konlpy = use_konlpy
        self.analyzer = None
        
        if use_konlpy:
            try:
                from konlpy.tag import Okt
                self.analyzer = Okt()
                print("✓ KoNLPy 로드 성공")
            except ImportError:
                print("⚠ KoNLPy 없음 - 자체 파서 사용")
                self.use_konlpy = False
    
    def parse(self, text: str) -> List[Morpheme]:
        """
        텍스트를 형태소로 분해
        
        Returns:
            List[Morpheme]: 형태소 리스트
        """
        if self.use_konlpy and self.analyzer:
            return self._parse_with_konlpy(text)
        else:
            return self._parse_with_rules(text)
    
    def _parse_with_konlpy(self, text: str) -> List[Morpheme]:
        """KoNLPy 기반 파싱"""
        morphemes = []
        pos_result = self.analyzer.pos(text, norm=True, stem=True)
        
        current_pos = 0
        for idx, (surface, pos) in enumerate(pos_result):
            # 원문에서 위치 찾기
            start = text.find(surface, current_pos)
            end = start + len(surface)
            
            morphemes.append(Morpheme(
                surface=surface,
                lemma=surface,  # Okt는 자동 정규화
                pos=pos,
                position=idx,
                original_index=(start, end)
            ))
            current_pos = end
        
        return morphemes
    
    def _parse_with_rules(self, text: str) -> List[Morpheme]:
        """
        자체 규칙 기반 파서
        - 간단한 패턴 매칭으로 조사/어미 분리
        """
        morphemes = []
        
        # 어절 단위로 분리
        words = text.split()
        current_pos = 0
        
        for word_idx, word in enumerate(words):
            # 조사 분리 패턴
            josa_patterns = [
                (r'(.+)(은|는|이|가|을|를|의|에|에서|로|부터|까지|도|만|조차|마저)$', 'JX'),
                (r'(.+)(와|과|하고|랑)$', 'JC'),
            ]
            
            # 어미 분리 패턴
            eomi_patterns = [
                (r'(.+)(다|네|구나|군|지|ㄴ다|는다)$', 'EF'),  # 종결
                (r'(.+)(지만|면서|아서|어서|니까|므로|고|ㄴ데|는데)$', 'EC'),  # 연결
                (r'(.+)(ㄴ|는|ㄹ|던|을)$', 'ETM'),  # 관형형
            ]
            
            matched = False
            
            # 조사 매칭
            for pattern, pos in josa_patterns:
                match = re.match(pattern, word)
                if match:
                    stem, josa = match.groups()
                    
                    # 어간 추가
                    morphemes.append(Morpheme(
                        surface=stem,
                        lemma=stem,
                        pos='NNG',  # 일반명사로 가정
                        position=len(morphemes),
                        original_index=(current_pos, current_pos + len(stem))
                    ))
                    
                    # 조사 추가
                    morphemes.append(Morpheme(
                        surface=josa,
                        lemma=josa,
                        pos=pos,
                        position=len(morphemes),
                        original_index=(current_pos + len(stem), current_pos + len(word))
                    ))
                    matched = True
                    break
            
            # 어미 매칭
            if not matched:
                for pattern, pos in eomi_patterns:
                    match = re.match(pattern, word)
                    if match:
                        stem, eomi = match.groups()
                        
                        morphemes.append(Morpheme(
                            surface=stem,
                            lemma=stem,
                            pos='VV',  # 동사로 가정
                            position=len(morphemes),
                            original_index=(current_pos, current_pos + len(stem))
                        ))
                        
                        morphemes.append(Morpheme(
                            surface=eomi,
                            lemma=eomi,
                            pos=pos,
                            position=len(morphemes),
                            original_index=(current_pos + len(stem), current_pos + len(word))
                        ))
                        matched = True
                        break
            
            # 매칭 안 되면 그대로 추가
            if not matched:
                morphemes.append(Morpheme(
                    surface=word,
                    lemma=word,
                    pos='NNG',
                    position=len(morphemes),
                    original_index=(current_pos, current_pos + len(word))
                ))
            
            current_pos += len(word) + 1  # 공백 포함
        
        return morphemes


# ============================================================================
# 조사 강도 사전 (Particle Intensity Dictionary)
# ============================================================================

@dataclass
class JosaIntensity:
    """조사의 감정 조절 정보"""
    intensity_modifier: float  # 강도 배율 (0.5 ~ 2.0)
    function: str             # 기능 (emphasis, contrast, etc.)
    direction: str            # 방향 (neutral, positive, negative)
    chord_progression: str    # 음악적 코드 진행


JOSA_INTENSITY_DICT = {
    # ═══════════════════════════════════════════════════════════
    # 강조 조사 - 감정 증폭
    # ═══════════════════════════════════════════════════════════
    '는': JosaIntensity(
        intensity_modifier=1.2,
        function='topic_emphasis',
        direction='neutral',
        chord_progression='I → V'  # 주제 제시
    ),
    '도': JosaIntensity(
        intensity_modifier=1.15,
        function='additive_emphasis',
        direction='positive',
        chord_progression='I → vi'  # 누적 효과
    ),
    '까지': JosaIntensity(
        intensity_modifier=1.3,
        function='extremity',
        direction='intensify',
        chord_progression='V → I'  # 극단으로
    ),
    '조차': JosaIntensity(
        intensity_modifier=1.4,
        function='surprise_extremity',
        direction='intensify',
        chord_progression='viidim → I'  # 놀라움
    ),
    '마저': JosaIntensity(
        intensity_modifier=1.35,
        function='final_addition',
        direction='intensify',
        chord_progression='IV → I'  # 마지막 추가
    ),
    
    # ═══════════════════════════════════════════════════════════
    # 대조 조사 - 감정 반전 신호
    # ═══════════════════════════════════════════════════════════
    '은': JosaIntensity(
        intensity_modifier=1.1,
        function='contrast_preparation',
        direction='neutral',
        chord_progression='I → IV'  # 대조 준비
    ),
    
    # ═══════════════════════════════════════════════════════════
    # 중립 조사 - 감정 유지
    # ═══════════════════════════════════════════════════════════
    '이': JosaIntensity(
        intensity_modifier=1.0,
        function='subject',
        direction='neutral',
        chord_progression='I'
    ),
    '가': JosaIntensity(
        intensity_modifier=1.0,
        function='subject',
        direction='neutral',
        chord_progression='I'
    ),
    '을': JosaIntensity(
        intensity_modifier=1.0,
        function='object',
        direction='neutral',
        chord_progression='V'
    ),
    '를': JosaIntensity(
        intensity_modifier=1.0,
        function='object',
        direction='neutral',
        chord_progression='V'
    ),
    
    # ═══════════════════════════════════════════════════════════
    # 완화 조사 - 감정 약화
    # ═══════════════════════════════════════════════════════════
    '만': JosaIntensity(
        intensity_modifier=0.8,
        function='limitation',
        direction='diminish',
        chord_progression='I → iii'  # 제한
    ),
    '뿐': JosaIntensity(
        intensity_modifier=0.75,
        function='only',
        direction='diminish',
        chord_progression='I → vi'  # 오직
    ),
    
    # ═══════════════════════════════════════════════════════════
    # 방향/위치 조사 - 공간적 이동
    # ═══════════════════════════════════════════════════════════
    '에': JosaIntensity(
        intensity_modifier=1.0,
        function='location',
        direction='neutral',
        chord_progression='ii → V'
    ),
    '에서': JosaIntensity(
        intensity_modifier=1.05,
        function='source',
        direction='neutral',
        chord_progression='vi → ii'
    ),
    '로': JosaIntensity(
        intensity_modifier=1.0,
        function='direction',
        direction='neutral',
        chord_progression='V → I'
    ),
    
    # ═══════════════════════════════════════════════════════════
    # 비교 조사 - 대조 구조
    # ═══════════════════════════════════════════════════════════
    '보다': JosaIntensity(
        intensity_modifier=1.2,
        function='comparison',
        direction='contrast',
        chord_progression='IV → V → I'
    ),
    '처럼': JosaIntensity(
        intensity_modifier=1.1,
        function='similarity',
        direction='neutral',
        chord_progression='I → IV → I'
    ),
}


# ============================================================================
# 어미 강도 사전 (Ending Intensity Dictionary)
# ============================================================================

@dataclass
class EomiIntensity:
    """어미의 감정 조절 정보"""
    intensity_modifier: float  # 강도 배율
    finality: float           # 종결도 (0.0 ~ 1.0)
    polarity: str             # 극성 (positive, negative, neutral)
    transition_type: Optional[str]  # 전환 유형 (contrast, causal, etc.)
    reverse_emotion: bool     # 감정 반전 여부
    tempo_change: str         # 템포 변화 (accelerando, ritardando)
    dynamic_mark: str         # 악상 기호 (sf, pp, crescendo)


EOMI_INTENSITY_DICT = {
    # ═══════════════════════════════════════════════════════════
    # 종결어미 - 확정도와 뉘앙스
    # ═══════════════════════════════════════════════════════════
    '-다': EomiIntensity(
        intensity_modifier=1.0,
        finality=1.0,
        polarity='neutral',
        transition_type=None,
        reverse_emotion=False,
        tempo_change='stable',
        dynamic_mark='♩ (완전종지)'
    ),
    '-네': EomiIntensity(
        intensity_modifier=1.2,
        finality=0.8,
        polarity='surprise',
        transition_type=None,
        reverse_emotion=False,
        tempo_change='slight_accel',
        dynamic_mark='mf (놀라움)'
    ),
    '-구나': EomiIntensity(
        intensity_modifier=1.3,
        finality=0.9,
        polarity='realization',
        transition_type=None,
        reverse_emotion=False,
        tempo_change='rubato',
        dynamic_mark='mp → f (깨달음)'
    ),
    '-군': EomiIntensity(
        intensity_modifier=1.25,
        finality=0.85,
        polarity='realization',
        transition_type=None,
        reverse_emotion=False,
        tempo_change='slight_rit',
        dynamic_mark='mf (인지)'
    ),
    '-ㄴ다': EomiIntensity(
        intensity_modifier=1.0,
        finality=1.0,
        polarity='neutral',
        transition_type=None,
        reverse_emotion=False,
        tempo_change='stable',
        dynamic_mark='♩'
    ),
    '-냐': EomiIntensity(
        intensity_modifier=1.1,
        finality=0.5,
        polarity='question',
        transition_type=None,
        reverse_emotion=False,
        tempo_change='rising',
        dynamic_mark='𝄐 (반종지)'
    ),
    '-자': EomiIntensity(
        intensity_modifier=1.15,
        finality=0.8,
        polarity='proposal',
        transition_type=None,
        reverse_emotion=False,
        tempo_change='accelerando',
        dynamic_mark='f (제안)'
    ),
    
    # ═══════════════════════════════════════════════════════════
    # 연결어미 - 감정 전환의 핵심! ⭐⭐⭐
    # ═══════════════════════════════════════════════════════════
    '-지만': EomiIntensity(
        intensity_modifier=1.1,
        finality=0.0,
        polarity='contrast',
        transition_type='contrast',
        reverse_emotion=True,  # 🔥 감정 반전!
        tempo_change='subito_change',
        dynamic_mark='sf → p (급격한 반전)'
    ),
    '-만': EomiIntensity(
        intensity_modifier=1.1,
        finality=0.0,
        polarity='contrast',
        transition_type='contrast',
        reverse_emotion=True,
        tempo_change='subito_change',
        dynamic_mark='sf (역접)'
    ),
    '-면서': EomiIntensity(
        intensity_modifier=1.0,
        finality=0.0,
        polarity='parallel',
        transition_type='parallel',
        reverse_emotion=False,
        tempo_change='maintain',
        dynamic_mark='≈ (동시 진행)'
    ),
    '-고': EomiIntensity(
        intensity_modifier=1.0,
        finality=0.0,
        polarity='sequential',
        transition_type='sequential',
        reverse_emotion=False,
        tempo_change='steady',
        dynamic_mark='→ (순차 진행)'
    ),
    '-아서': EomiIntensity(
        intensity_modifier=1.05,
        finality=0.0,
        polarity='causal',
        transition_type='causal',
        reverse_emotion=False,
        tempo_change='logical_flow',
        dynamic_mark='→ (인과)'
    ),
    '-어서': EomiIntensity(
        intensity_modifier=1.05,
        finality=0.0,
        polarity='causal',
        transition_type='causal',
        reverse_emotion=False,
        tempo_change='logical_flow',
        dynamic_mark='→ (인과)'
    ),
    '-니까': EomiIntensity(
        intensity_modifier=1.1,
        finality=0.0,
        polarity='causal_strong',
        transition_type='causal',
        reverse_emotion=False,
        tempo_change='emphatic',
        dynamic_mark='f → mf (강한 인과)'
    ),
    '-므로': EomiIntensity(
        intensity_modifier=1.15,
        finality=0.0,
        polarity='causal_formal',
        transition_type='causal',
        reverse_emotion=False,
        tempo_change='formal',
        dynamic_mark='mp (격식)'
    ),
    '-ㄴ데': EomiIntensity(
        intensity_modifier=1.05,
        finality=0.0,
        polarity='background',
        transition_type='background',
        reverse_emotion=False,
        tempo_change='soft',
        dynamic_mark='p (배경 제시)'
    ),
    '-는데': EomiIntensity(
        intensity_modifier=1.05,
        finality=0.0,
        polarity='background',
        transition_type='background',
        reverse_emotion=False,
        tempo_change='soft',
        dynamic_mark='p (배경)'
    ),
    
    # ═══════════════════════════════════════════════════════════
    # 전성어미 - 수식 강화
    # ═══════════════════════════════════════════════════════════
    '-ㄴ': EomiIntensity(
        intensity_modifier=1.1,
        finality=0.0,
        polarity='modifier',
        transition_type='modification',
        reverse_emotion=False,
        tempo_change='sustained',
        dynamic_mark='tenuto (지속)'
    ),
    '-는': EomiIntensity(
        intensity_modifier=1.15,
        finality=0.0,
        polarity='progressive',
        transition_type='modification',
        reverse_emotion=False,
        tempo_change='progressive',
        dynamic_mark='crescendo (진행 중)'
    ),
    '-ㄹ': EomiIntensity(
        intensity_modifier=1.1,
        finality=0.0,
        polarity='future',
        transition_type='modification',
        reverse_emotion=False,
        tempo_change='anticipation',
        dynamic_mark='anticipato (예상)'
    ),
    
    # ═══════════════════════════════════════════════════════════
    # 시제어미 - 시간감
    # ═══════════════════════════════════════════════════════════
    '-었-': EomiIntensity(
        intensity_modifier=1.0,
        finality=0.0,
        polarity='past',
        transition_type='temporal',
        reverse_emotion=False,
        tempo_change='ritardando',
        dynamic_mark='rit. (과거로)'
    ),
    '-았-': EomiIntensity(
        intensity_modifier=1.0,
        finality=0.0,
        polarity='past',
        transition_type='temporal',
        reverse_emotion=False,
        tempo_change='ritardando',
        dynamic_mark='rit. (과거로)'
    ),
    '-겠-': EomiIntensity(
        intensity_modifier=1.05,
        finality=0.0,
        polarity='future',
        transition_type='temporal',
        reverse_emotion=False,
        tempo_change='accelerando',
        dynamic_mark='accel. (미래로)'
    ),
}


# ============================================================================
# 인터넷 표현 강도 사전
# ============================================================================

@dataclass
class InternetSlangIntensity:
    """인터넷 표현의 감정 정보"""
    emotion_type: str         # 감정 유형
    intensity: float          # 강도
    authenticity: float       # 진정성 (0.0 ~ 1.0)


INTERNET_SLANG_DICT = {
    # ═══════════════════════════════════════════════════════════
    # 자음 반복 - 감정 증폭
    # ═══════════════════════════════════════════════════════════
    'ㅋ': InternetSlangIntensity('amusement', 1.2, 0.9),
    'ㅋㅋ': InternetSlangIntensity('amusement', 1.4, 0.8),
    'ㅋㅋㅋ': InternetSlangIntensity('amusement', 1.6, 0.7),
    'ㅋㅋㅋㅋ': InternetSlangIntensity('amusement', 1.8, 0.5),
    'ㅋㅋㅋㅋㅋ': InternetSlangIntensity('amusement', 2.0, 0.3),  # 과장
    
    'ㅎ': InternetSlangIntensity('joy', 1.1, 0.95),
    'ㅎㅎ': InternetSlangIntensity('joy', 1.3, 0.85),
    'ㅎㅎㅎ': InternetSlangIntensity('joy', 1.5, 0.7),
    
    'ㅠ': InternetSlangIntensity('sadness', 1.3, 0.9),
    'ㅠㅠ': InternetSlangIntensity('sadness', 1.6, 0.85),
    'ㅠㅠㅠ': InternetSlangIntensity('sadness', 1.9, 0.8),
    'ㅜㅜ': InternetSlangIntensity('sadness', 1.6, 0.85),
    
    # ═══════════════════════════════════════════════════════════
    # 특수 기호 - 뉘앙스 변화
    # ═══════════════════════════════════════════════════════════
    '...': InternetSlangIntensity('contemplation', 0.7, 0.9),
    '..': InternetSlangIntensity('hesitation', 0.8, 0.85),
    '!': InternetSlangIntensity('excitement', 1.3, 0.9),
    '!!': InternetSlangIntensity('excitement', 1.6, 0.85),
    '!!!': InternetSlangIntensity('excitement', 1.9, 0.7),
    '?': InternetSlangIntensity('confusion', 1.1, 0.9),
    '??': InternetSlangIntensity('confusion', 1.4, 0.85),
    '???': InternetSlangIntensity('confusion', 1.7, 0.8),
    
    # ═══════════════════════════════════════════════════════════
    # 인터넷 신조어
    # ═══════════════════════════════════════════════════════════
    'ㄹㅇ': InternetSlangIntensity('emphasis', 1.3, 0.8),  # 레알
    'ㅇㅈ': InternetSlangIntensity('agreement', 1.2, 0.8),  # 인정
    'ㅇㅋ': InternetSlangIntensity('acceptance', 1.1, 0.85),  # 오케이
    'ㄱㅅ': InternetSlangIntensity('gratitude', 1.2, 0.9),  # 감사
    'ㅈㅅ': InternetSlangIntensity('apology', 1.2, 0.85),  # 죄송
}


# ============================================================================
# 형태소 강도 적용 엔진
# ============================================================================

class MorphemeIntensityEngine:
    """
    형태소별 강도를 계산하고 감정에 적용
    """
    
    def __init__(self):
        self.josa_dict = JOSA_INTENSITY_DICT
        self.eomi_dict = EOMI_INTENSITY_DICT
        self.slang_dict = INTERNET_SLANG_DICT
    
    def calculate_morpheme_intensity(
        self, 
        morphemes: List[Morpheme],
        base_emotion: Dict[str, float]
    ) -> Dict[str, any]:
        """
        형태소 리스트에서 감정 강도 조절 정보 추출
        
        Returns:
            {
                'intensity_modifiers': List[float],  # 각 형태소의 강도 배율
                'emotion_transitions': List[dict],    # 감정 전환 정보
                'musical_articulations': List[str],   # 악상 기호
                'final_intensity': float              # 최종 강도
            }
        """
        result = {
            'intensity_modifiers': [],
            'emotion_transitions': [],
            'musical_articulations': [],
            'internet_expressions': [],
            'chord_progressions': []
        }
        
        cumulative_intensity = 1.0
        current_emotion = base_emotion.copy()
        
        for morph in morphemes:
            # 조사 처리
            if morph.pos in ['JX', 'JC']:
                josa_info = self.josa_dict.get(morph.surface)
                if josa_info:
                    result['intensity_modifiers'].append({
                        'morpheme': morph.surface,
                        'type': 'josa',
                        'modifier': josa_info.intensity_modifier,
                        'function': josa_info.function
                    })
                    result['chord_progressions'].append(
                        josa_info.chord_progression
                    )
                    cumulative_intensity *= josa_info.intensity_modifier
            
            # 어미 처리
            elif morph.pos in ['EF', 'EC', 'ETM']:
                eomi_info = self.eomi_dict.get(morph.surface)
                if eomi_info:
                    result['intensity_modifiers'].append({
                        'morpheme': morph.surface,
                        'type': 'eomi',
                        'modifier': eomi_info.intensity_modifier,
                        'finality': eomi_info.finality
                    })
                    result['musical_articulations'].append(
                        eomi_info.dynamic_mark
                    )
                    
                    cumulative_intensity *= eomi_info.intensity_modifier
                    
                    # 감정 전환 감지 (핵심!)
                    if eomi_info.reverse_emotion:
                        result['emotion_transitions'].append({
                            'position': morph.position,
                            'type': 'reversal',
                            'trigger': morph.surface,
                            'before_emotion': current_emotion.copy(),
                            'transition_type': eomi_info.transition_type
                        })
                        # 감정 반전 처리는 상위 레이어에서 수행
        
        result['final_intensity'] = cumulative_intensity
        
        return result
    
    def detect_internet_expressions(self, text: str) -> List[Dict]:
        """
        인터넷 표현 감지
        """
        expressions = []
        
        for pattern, intensity_info in self.slang_dict.items():
            if pattern in text:
                count = text.count(pattern)
                expressions.append({
                    'pattern': pattern,
                    'count': count,
                    'emotion': intensity_info.emotion_type,
                    'intensity': intensity_info.intensity,
                    'authenticity': intensity_info.authenticity
                })
        
        return expressions


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    # 형태소 분석기 초기화
    analyzer = MorphemeAnalyzer(use_konlpy=False)  # 자체 파서 사용
    intensity_engine = MorphemeIntensityEngine()
    
    # 테스트 문장
    test_sentences = [
        "정말 기쁘네요!",
        "슬프지만 견뎌야 해",
        "너무 짜증나ㅋㅋㅋ",
        "친구는 좋은데 날씨가 별로다",
        "아 진짜 짜증나넼ㅋㅋ"
    ]
    
    print("=" * 60)
    print("형태소 강도 분석 시스템 테스트")
    print("=" * 60)
    
    for sent in test_sentences:
        print(f"\n📝 문장: {sent}")
        print("-" * 60)
        
        # 형태소 분석
        morphemes = analyzer.parse(sent)
        
        print("형태소:")
        for m in morphemes:
            print(f"  {m.surface} ({m.pos})")
        
        # 강도 계산
        base_emotion = {'joy': 0.5, 'sadness': 0.3}
        intensity_result = intensity_engine.calculate_morpheme_intensity(
            morphemes, base_emotion
        )
        
        print(f"\n최종 강도 배율: {intensity_result['final_intensity']:.2f}")
        
        if intensity_result['intensity_modifiers']:
            print("\n강도 조절자:")
            for mod in intensity_result['intensity_modifiers']:
                print(f"  {mod['morpheme']} ({mod['type']}): "
                      f"×{mod['modifier']:.2f}")
        
        if intensity_result['emotion_transitions']:
            print("\n⚠️  감정 전환 감지:")
            for trans in intensity_result['emotion_transitions']:
                print(f"  위치 {trans['position']}: "
                      f"{trans['trigger']} → {trans['type']}")
        
        # 인터넷 표현 감지
        internet_expr = intensity_engine.detect_internet_expressions(sent)
        if internet_expr:
            print("\n💬 인터넷 표현:")
            for expr in internet_expr:
                print(f"  {expr['pattern']}: "
                      f"{expr['emotion']} (×{expr['intensity']:.1f})")
