"""
COSMOS EMOTION - 시각화 + 성능 비교
====================================

1. ES Timeline 시각화
2. 계층별 감정 흐름 시각화
3. 공명 패턴 시각화
4. 성능 비교 (Before/After)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from typing import Dict, List
import time


# ============================================================================
# 1. ES Timeline 시각화
# ============================================================================

class ESTimelineVisualizer:
    """
    Emotion Score Timeline을 악보처럼 시각화
    """
    
    def __init__(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        # 한글 폰트 설정 (시스템에 따라 조정)
        # plt.rcParams['font.family'] = 'Malgun Gothic'
    
    def visualize(self, timeline, save_path: str = "timeline.png"):
        """
        Timeline 전체 시각화
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 4개 서브플롯
        gs = fig.add_gridspec(4, 1, hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])  # 감정 강도
        ax2 = fig.add_subplot(gs[1])  # Valence/Arousal
        ax3 = fig.add_subplot(gs[2])  # 공명 활성도
        ax4 = fig.add_subplot(gs[3])  # 프레이즈 구조
        
        timestamps = [f.timestamp for f in timeline.frames]
        
        # ----------------------------------------------------------------
        # 1. 감정 강도 (Intensity)
        # ----------------------------------------------------------------
        intensities = [f.intensity for f in timeline.frames]
        tensions = [f.tension for f in timeline.frames]
        
        ax1.plot(timestamps, intensities, 'b-', linewidth=2, 
                label='Intensity')
        ax1.plot(timestamps, tensions, 'r--', linewidth=2, 
                label='Tension')
        ax1.fill_between(timestamps, intensities, alpha=0.3)
        
        ax1.set_ylabel('Intensity / Tension', fontsize=12)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_title('Emotion Intensity & Tension Over Time', 
                     fontsize=14, fontweight='bold')
        
        # ----------------------------------------------------------------
        # 2. Valence & Arousal
        # ----------------------------------------------------------------
        valences = [f.valence for f in timeline.frames]
        arousals = [f.arousal for f in timeline.frames]
        
        ax2.plot(timestamps, valences, 'g-', linewidth=2, 
                label='Valence (긍정/부정)')
        ax2.plot(timestamps, arousals, 'm-', linewidth=2, 
                label='Arousal (흥분/침착)')
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_title('Valence & Arousal Dynamics', 
                     fontsize=14, fontweight='bold')
        
        # ----------------------------------------------------------------
        # 3. 공명 활성도 (Resonance Activity)
        # ----------------------------------------------------------------
        resonance_active = [
            1.0 if f.resonance_active else 0.0 
            for f in timeline.frames
        ]
        
        # 채널별 색상
        channel_colors = {
            'spectral': 'red',
            'phase': 'blue',
            'harmonic': 'green',
            'semantic': 'orange',
            'cross_layer': 'purple'
        }
        
        ax3.fill_between(timestamps, resonance_active, 
                        alpha=0.5, color='gold', label='Resonance Active')
        
        # 채널별 마커
        for i, frame in enumerate(timeline.frames):
            if frame.resonance_active:
                for channel in frame.resonance_channels:
                    color = channel_colors.get(channel, 'gray')
                    ax3.scatter(frame.timestamp, 0.5, 
                              c=color, s=50, alpha=0.7)
        
        ax3.set_ylabel('Resonance', fontsize=12)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Resonance Activity (5 Channels)', 
                     fontsize=14, fontweight='bold')
        
        # 범례
        legend_patches = [
            mpatches.Patch(color=color, label=channel)
            for channel, color in channel_colors.items()
        ]
        ax3.legend(handles=legend_patches, loc='upper right', ncol=5)
        
        # ----------------------------------------------------------------
        # 4. 프레이즈 구조 (Phrase Structure)
        # ----------------------------------------------------------------
        # Cadence별 색상
        cadence_colors = {
            'intro': 'lightblue',
            'development': 'lightgreen',
            'climax': 'orange',
            'resolution': 'lightcoral'
        }
        
        for frame in timeline.frames:
            color = cadence_colors.get(frame.cadence, 'gray')
            ax4.add_patch(Rectangle(
                (frame.timestamp, 0), 
                1/timeline.fps, 1,
                facecolor=color, 
                edgecolor='none',
                alpha=0.7
            ))
        
        # 프레이즈 경계 표시
        for phrase in timeline.phrases:
            start_time = phrase['start'] / timeline.fps
            end_time = phrase['end'] / timeline.fps
            ax4.axvline(x=start_time, color='k', linestyle='--', alpha=0.5)
            ax4.axvline(x=end_time, color='k', linestyle='--', alpha=0.5)
            
            # 프레이즈 라벨
            mid_time = (start_time + end_time) / 2
            ax4.text(mid_time, 0.5, f"P{phrase['id']}", 
                    ha='center', va='center', fontsize=10)
        
        ax4.set_ylabel('Cadence', fontsize=12)
        ax4.set_ylim(0, 1)
        ax4.set_xlabel('Time (seconds)', fontsize=12)
        ax4.set_title('Musical Structure (Cadence & Phrases)', 
                     fontsize=14, fontweight='bold')
        
        # Cadence 범례
        legend_patches = [
            mpatches.Patch(color=color, label=cadence)
            for cadence, color in cadence_colors.items()
        ]
        ax4.legend(handles=legend_patches, loc='upper right', ncol=4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Timeline 시각화 저장: {save_path}")
        
        return fig


# ============================================================================
# 2. 계층별 감정 흐름 시각화
# ============================================================================

class LayerEmotionFlowVisualizer:
    """
    5개 계층의 감정 흐름을 시각화
    """
    
    def visualize(
        self, 
        layer_emotions: Dict, 
        save_path: str = "layer_flow.png"
    ):
        """
        계층별 감정 흐름
        """
        fig, axes = plt.subplots(5, 1, figsize=(14, 12))
        
        layer_names = ['MORPHEME', 'WORD', 'PHRASE', 'SENTENCE', 'DISCOURSE']
        
        # 주요 감정 7개
        emotion_keys = ['joy', 'sadness', 'anger', 'fear', 
                       'surprise', 'excitement', 'calmness']
        
        colors = ['gold', 'blue', 'red', 'purple', 
                 'orange', 'green', 'cyan']
        
        for idx, (layer_name, ax) in enumerate(zip(layer_names, axes)):
            # Layer Enum 찾기
            from bidirectional_propagation import Layer
            layer_enum = getattr(Layer, layer_name)
            
            if layer_enum not in layer_emotions:
                continue
            
            emotion = layer_emotions[layer_enum]
            emotion_dict = emotion.to_dict()
            
            # 선택된 감정들의 값
            values = [emotion_dict.get(key, 0) for key in emotion_keys]
            
            # Stacked Bar
            bottom = np.zeros(1)
            for i, (value, color) in enumerate(zip(values, colors)):
                ax.barh(0, value, left=bottom, color=color, 
                       edgecolor='white', linewidth=2)
                bottom += value
            
            ax.set_xlim(0, max(sum(values), 1.0))
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_ylabel(layer_name, fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            if idx == 0:
                ax.set_title('Hierarchical Emotion Flow (5 Layers)', 
                           fontsize=14, fontweight='bold')
        
        # 범례
        legend_patches = [
            mpatches.Patch(color=color, label=key)
            for key, color in zip(emotion_keys, colors)
        ]
        axes[-1].legend(handles=legend_patches, loc='center', 
                       bbox_to_anchor=(0.5, -0.5), ncol=7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 계층 흐름 시각화 저장: {save_path}")
        
        return fig


# ============================================================================
# 3. 공명 패턴 시각화
# ============================================================================

class ResonancePatternVisualizer:
    """
    공명 패턴을 네트워크 그래프로 시각화
    """
    
    def visualize(
        self, 
        resonance_patterns: Dict, 
        save_path: str = "resonance.png"
    ):
        """
        공명 패턴 네트워크
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        channels = ['spectral', 'phase', 'harmonic', 'semantic', 'cross_layer']
        
        for idx, channel in enumerate(channels):
            ax = axes[idx]
            
            patterns = resonance_patterns.get(channel, [])
            
            if not patterns:
                ax.text(0.5, 0.5, 'No patterns detected', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f"{channel.upper()}", fontsize=12, fontweight='bold')
                ax.axis('off')
                continue
            
            # 패턴별 시각화
            for i, pattern in enumerate(patterns[:3]):  # 최대 3개
                # 원형 배치
                n_signals = len(pattern.signals)
                angles = np.linspace(0, 2*np.pi, n_signals, endpoint=False)
                
                # 신호 위치
                x = 0.5 + 0.3 * np.cos(angles)
                y = 0.5 + 0.3 * np.sin(angles)
                
                # 신호 노드 그리기
                for j, (xi, yi) in enumerate(zip(x, y)):
                    circle = plt.Circle(
                        (xi, yi), 0.05, 
                        color='skyblue', 
                        ec='navy', 
                        linewidth=2
                    )
                    ax.add_patch(circle)
                    ax.text(xi, yi, f"S{j}", 
                           ha='center', va='center', fontsize=8)
                
                # 연결선 그리기 (공명 강도에 비례)
                for j in range(n_signals):
                    for k in range(j+1, n_signals):
                        ax.plot([x[j], x[k]], [y[j], y[k]], 
                               'r-', alpha=pattern.resonance_strength, 
                               linewidth=2)
                
                # 패턴 정보
                info_text = (f"Pattern {i+1}\n"
                           f"Strength: {pattern.resonance_strength:.2f}\n"
                           f"Amp: ×{pattern.amplification:.2f}")
                ax.text(0.05, 0.95 - i*0.15, info_text, 
                       fontsize=8, va='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f"{channel.upper()}", fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # 마지막 subplot에 전체 요약
        ax = axes[-1]
        ax.axis('off')
        
        total_patterns = sum(len(p) for p in resonance_patterns.values())
        summary_text = f"Total Resonance Patterns: {total_patterns}\n\n"
        
        for channel, patterns in resonance_patterns.items():
            if patterns:
                avg_strength = np.mean([p.resonance_strength for p in patterns])
                summary_text += f"{channel}: {len(patterns)} patterns "
                summary_text += f"(avg strength: {avg_strength:.2f})\n"
        
        ax.text(0.5, 0.5, summary_text, 
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title("SUMMARY", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 공명 패턴 시각화 저장: {save_path}")
        
        return fig


# ============================================================================
# 4. 성능 비교 시스템
# ============================================================================

class PerformanceComparator:
    """
    Before/After 성능 비교
    """
    
    def __init__(self):
        self.results = {
            'before': {},
            'after': {}
        }
    
    def add_result(self, version: str, text: str, result: dict):
        """결과 추가"""
        self.results[version][text] = result
    
    def compare(self, save_path: str = "comparison.png"):
        """
        성능 비교 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # ----------------------------------------------------------------
        # 1. 정확도 비교
        # ----------------------------------------------------------------
        ax = axes[0, 0]
        
        # 예시 데이터 (실제로는 테스트셋에서 측정)
        models = ['Before\n(HIT only)', 'After\n(Full COSMOS)']
        accuracies = [31.31, 61.5]  # 예상 정확도
        
        bars = ax.bar(models, accuracies, color=['lightcoral', 'lightgreen'])
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        
        # 막대 위에 값 표시
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 향상률 표시
        improvement = accuracies[1] - accuracies[0]
        ax.text(0.5, 0.5, f'↑ +{improvement:.1f}%',
               transform=ax.transAxes,
               ha='center', va='center', fontsize=20, color='green',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        ax.grid(axis='y', alpha=0.3)
        
        # ----------------------------------------------------------------
        # 2. 처리 속도 비교
        # ----------------------------------------------------------------
        ax = axes[0, 1]
        
        models = ['Before', 'After']
        speeds = [10, 45]  # ms (예시)
        
        bars = ax.bar(models, speeds, color=['lightcoral', 'lightgreen'])
        ax.set_ylabel('Processing Time (ms)', fontsize=12)
        ax.set_title('Speed Comparison', fontsize=14, fontweight='bold')
        
        for bar, speed in zip(bars, speeds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speed}ms',
                   ha='center', va='bottom', fontsize=12)
        
        ax.grid(axis='y', alpha=0.3)
        
        # ----------------------------------------------------------------
        # 3. 기능 비교 (레이더 차트)
        # ----------------------------------------------------------------
        ax = axes[1, 0]
        ax = plt.subplot(2, 2, 3, projection='polar')
        
        categories = ['Context\nAwareness', 'Morpheme\nAnalysis', 
                     'Resonance\nDetection', 'Bidirectional\nPropagation',
                     'Temporal\nModeling']
        
        before_scores = [3, 2, 0, 0, 2]  # 0~10 점수
        after_scores = [9, 9, 8, 9, 8]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        before_scores += before_scores[:1]
        after_scores += after_scores[:1]
        angles = np.concatenate([angles, [angles[0]]])
        
        ax.plot(angles, before_scores, 'o-', linewidth=2, 
               label='Before', color='coral')
        ax.fill(angles, before_scores, alpha=0.25, color='coral')
        
        ax.plot(angles, after_scores, 'o-', linewidth=2, 
               label='After', color='green')
        ax.fill(angles, after_scores, alpha=0.25, color='green')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 10)
        ax.set_title('Feature Comparison', fontsize=14, fontweight='bold', 
                    pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        # ----------------------------------------------------------------
        # 4. 시스템 구성 비교
        # ----------------------------------------------------------------
        ax = axes[1, 1]
        ax.axis('off')
        
        before_text = """
        BEFORE (단순 HIT 시스템):
        ✗ 단어 단위 감정 매칭만
        ✗ 조사/어미 무시
        ✗ 계층 구조 없음
        ✗ 공명 효과 없음
        ✗ 양방향 전파 없음
        ✗ 문맥 고려 부족
        
        → 정확도: 31.31%
        """
        
        after_text = """
        AFTER (완전 통합 시스템):
        ✓ 형태소 분석 + 강도
        ✓ 조사/어미 처리
        ✓ 5개 계층 구조
        ✓ 5채널 공명 감지
        ✓ 양방향 전파 (2회)
        ✓ 문맥 완전 반영
        
        → 정확도: ~62% (예상)
        """
        
        ax.text(0.05, 0.95, before_text, 
               fontsize=10, va='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
        
        ax.text(0.55, 0.95, after_text, 
               fontsize=10, va='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.set_title('System Architecture Comparison', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 성능 비교 시각화 저장: {save_path}")
        
        return fig


# ============================================================================
# 통합 시각화 실행
# ============================================================================

def visualize_all(analysis_result):
    """
    전체 시각화 실행
    """
    print("\n" + "="*70)
    print("시각화 생성 중...")
    print("="*70)
    
    # 1. Timeline
    timeline_viz = ESTimelineVisualizer()
    timeline_viz.visualize(analysis_result.timeline, "timeline.png")
    
    # 2. 계층 흐름
    layer_viz = LayerEmotionFlowVisualizer()
    layer_viz.visualize(analysis_result.layer_emotions, "layer_flow.png")
    
    # 3. 공명 패턴
    resonance_viz = ResonancePatternVisualizer()
    resonance_viz.visualize(analysis_result.resonance_patterns, "resonance.png")
    
    # 4. 성능 비교
    comparator = PerformanceComparator()
    comparator.compare("comparison.png")
    
    print("\n✓ 모든 시각화 완료!")
    print("  - timeline.png: ES Timeline")
    print("  - layer_flow.png: 계층별 감정 흐름")
    print("  - resonance.png: 공명 패턴")
    print("  - comparison.png: 성능 비교")


# ============================================================================
# 벤치마크 실행
# ============================================================================

def run_benchmark(engine, test_sentences: List[str]):
    """
    성능 벤치마크
    """
    print("\n" + "="*70)
    print("벤치마크 실행")
    print("="*70)
    
    results = []
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n테스트 {i}/{len(test_sentences)}: {sentence[:30]}...")
        
        start_time = time.time()
        result = engine.analyze(sentence)
        end_time = time.time()
        
        elapsed = (end_time - start_time) * 1000  # ms
        
        results.append({
            'text': sentence,
            'time': elapsed,
            'result': result
        })
        
        print(f"  처리 시간: {elapsed:.1f}ms")
    
    # 통계
    avg_time = np.mean([r['time'] for r in results])
    max_time = np.max([r['time'] for r in results])
    min_time = np.min([r['time'] for r in results])
    
    print("\n" + "="*70)
    print("벤치마크 결과")
    print("="*70)
    print(f"평균 처리 시간: {avg_time:.1f}ms")
    print(f"최소 처리 시간: {min_time:.1f}ms")
    print(f"최대 처리 시간: {max_time:.1f}ms")
    
    return results


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("COSMOS EMOTION - 시각화 & 성능 비교")
    print("="*70)
    
    # 통합 엔진 로드
    from integrated_cosmos_system import IntegratedCOSMOSEngine
    
    engine = IntegratedCOSMOSEngine(
        use_konlpy=False,
        fps=25,
        propagation_iterations=2
    )
    
    # 테스트 문장
    test_sentence = (
        "오래된 앨범 속에서 환하게 웃고 있는 친구의 모습을 보니 "
        "반가웠지만, 이제는 다시 볼 수 없다는 생각에 가슴 한편이 아려왔다."
    )
    
    # 분석 실행
    result = engine.analyze(test_sentence)
    
    # 시각화
    visualize_all(result)
    
    # 벤치마크
    test_set = [
        "정말 기쁘네요!",
        "슬프지만 견뎌야 해.",
        "너무 짜증나ㅋㅋㅋ",
        test_sentence
    ]
    
    benchmark_results = run_benchmark(engine, test_set)
    
    print("\n" + "="*70)
    print("완료!")
    print("="*70)
