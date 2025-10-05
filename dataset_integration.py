"""
COSMOS EMOTION - 데이터셋 통합 + 성능 평가
============================================

기능:
1. AIHub 데이터셋 로드 및 전처리
2. Train/Val/Test 분할
3. 모델 학습 (가중치 조정)
4. 성능 평가 (정확도, F1, 혼동 행렬)
5. Before/After 비교
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# 1. 데이터셋 로더
# ============================================================================

class DatasetLoader:
    """
    AIHub 감정 데이터셋 로드
    
    지원 형식:
    - 4차년도, 5차년도, 5차년도_2차
    - JSON 형식
    - WAV 파일 매핑
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.emotion_map = self._create_emotion_map()
    
    def _create_emotion_map(self) -> Dict:
        """
        데이터셋 감정 라벨 → COSMOS 28차원 매핑
        """
        return {
            # 기본 감정
            'joy': ['joy', 'happy', 'pleased', 'delighted'],
            'sadness': ['sad', 'unhappy', 'sorrowful', 'depressed'],
            'anger': ['angry', 'mad', 'furious', 'irritated'],
            'fear': ['afraid', 'scared', 'fearful', 'anxious'],
            'disgust': ['disgusted', 'repulsed', 'revolted'],
            'surprise': ['surprised', 'amazed', 'astonished'],
            
            # 복합 감정
            'excitement': ['excited', 'thrilled', 'enthusiastic'],
            'calmness': ['calm', 'peaceful', 'serene'],
            'disappointment': ['disappointed', 'let down'],
            'empathic_pain': ['empathetic', 'sympathetic'],
            'amusement': ['amused', 'entertained'],
            'confusion': ['confused', 'bewildered'],
            
            # 한국 감정
            'han': ['한', 'resentment'],
            'jeong': ['정', 'affection'],
            'nunchi': ['눈치', 'social awareness'],
            'hyeontta': ['혀따닥', 'tsk'],
            'menboong': ['멘붕', 'mental breakdown'],
        }
    
    def load_dataset(
        self, 
        dataset_name: str = "5차년도"
    ) -> List[Dict]:
        """
        데이터셋 로드
        
        Returns:
            [
                {
                    'text': str,
                    'emotion': str,
                    'intensity': float,
                    'wav_file': str (optional)
                },
                ...
            ]
        """
        dataset_path = os.path.join(self.data_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"⚠ 데이터셋을 찾을 수 없습니다: {dataset_path}")
            return self._load_sample_dataset()
        
        samples = []
        
        # JSON 파일 탐색
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # 샘플 추출
                            sample = self._extract_sample(data)
                            if sample:
                                samples.append(sample)
                    
                    except Exception as e:
                        print(f"⚠ 파일 로드 실패: {file_path} - {e}")
        
        print(f"✓ {len(samples)}개 샘플 로드 완료")
        return samples
    
    def _extract_sample(self, data: Dict) -> Dict:
        """
        JSON에서 샘플 추출
        
        AIHub 데이터 구조:
        {
            "id": "...",
            "text": "...",
            "emotion": {
                "category": "joy",
                "intensity": 0.8
            },
            "metadata": {
                "wav_file": "..."
            }
        }
        """
        try:
            return {
                'text': data.get('text', ''),
                'emotion': data.get('emotion', {}).get('category', 'neutral'),
                'intensity': data.get('emotion', {}).get('intensity', 0.5),
                'wav_file': data.get('metadata', {}).get('wav_file', None)
            }
        except Exception as e:
            return None
    
    def _load_sample_dataset(self) -> List[Dict]:
        """
        샘플 데이터셋 (테스트용)
        """
        print("⚠ 실제 데이터셋 대신 샘플 데이터 사용")
        
        samples = [
            {'text': '정말 기쁘네요!', 'emotion': 'joy', 'intensity': 0.9},
            {'text': '너무 슬프다...', 'emotion': 'sadness', 'intensity': 0.8},
            {'text': '화가 나네', 'emotion': 'anger', 'intensity': 0.7},
            {'text': '무섭다', 'emotion': 'fear', 'intensity': 0.8},
            {'text': '짜증나ㅋㅋㅋ', 'emotion': 'anger', 'intensity': 0.6},
            {'text': '슬프지만 견뎌야 해', 'emotion': 'sadness', 'intensity': 0.7},
            {'text': '오늘은 최고의 날!', 'emotion': 'joy', 'intensity': 1.0},
            {'text': '실망했어', 'emotion': 'disappointment', 'intensity': 0.6},
        ] * 50  # 400개로 확장
        
        return samples


# ============================================================================
# 2. 데이터 전처리
# ============================================================================

class DataPreprocessor:
    """
    데이터 전처리 및 증강
    """
    
    def __init__(self):
        self.label_encoder = {}
        self.label_decoder = {}
    
    def preprocess(
        self, 
        samples: List[Dict],
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple:
        """
        데이터 전처리 및 분할
        
        Returns:
            (train_data, val_data, test_data)
        """
        # 라벨 인코딩
        self._create_label_encoding(samples)
        
        # 텍스트 정제
        cleaned_samples = [
            self._clean_sample(s) for s in samples
        ]
        
        # 필터링 (빈 텍스트 제거)
        cleaned_samples = [
            s for s in cleaned_samples 
            if s['text'].strip()
        ]
        
        # Train/Val/Test 분할
        train_val, test = train_test_split(
            cleaned_samples,
            test_size=test_size,
            stratify=[s['emotion'] for s in cleaned_samples],
            random_state=42
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=[s['emotion'] for s in train_val],
            random_state=42
        )
        
        print(f"✓ 데이터 분할 완료:")
        print(f"  Train: {len(train)}개")
        print(f"  Val:   {len(val)}개")
        print(f"  Test:  {len(test)}개")
        
        return train, val, test
    
    def _create_label_encoding(self, samples: List[Dict]):
        """
        라벨 인코딩 생성
        """
        unique_labels = sorted(set(s['emotion'] for s in samples))
        
        self.label_encoder = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        self.label_decoder = {
            idx: label for label, idx in self.label_encoder.items()
        }
    
    def _clean_sample(self, sample: Dict) -> Dict:
        """
        샘플 정제
        """
        text = sample['text']
        
        # 불필요한 공백 제거
        text = ' '.join(text.split())
        
        # 특수 문자 정리 (일부만)
        # text = re.sub(r'[^\w\s가-힣.,!?]', '', text)
        
        sample['text'] = text
        return sample


# ============================================================================
# 3. 성능 평가기
# ============================================================================

class PerformanceEvaluator:
    """
    모델 성능 평가
    """
    
    def __init__(self, label_decoder: Dict):
        self.label_decoder = label_decoder
    
    def evaluate(
        self,
        engine,
        test_data: List[Dict],
        verbose: bool = True
    ) -> Dict:
        """
        모델 평가
        
        Returns:
            {
                'accuracy': float,
                'f1_score': float,
                'confusion_matrix': np.ndarray,
                'per_class_accuracy': Dict,
                'predictions': List
            }
        """
        if verbose:
            print("\n" + "="*70)
            print("성능 평가 시작")
            print("="*70)
        
        y_true = []
        y_pred = []
        predictions = []
        
        for i, sample in enumerate(test_data):
            if verbose and (i + 1) % 50 == 0:
                print(f"  진행: {i+1}/{len(test_data)}")
            
            # 분석
            result = engine.analyze(sample['text'])
            
            # 예측 감정 추출
            predicted_emotion = self._extract_predicted_emotion(result)
            
            y_true.append(sample['emotion'])
            y_pred.append(predicted_emotion)
            
            predictions.append({
                'text': sample['text'],
                'true': sample['emotion'],
                'predicted': predicted_emotion,
                'correct': sample['emotion'] == predicted_emotion
            })
        
        # 지표 계산
        accuracy = accuracy_score(y_true, y_pred)
        
        # F1 Score (macro)
        unique_labels = sorted(set(y_true + y_pred))
        f1 = f1_score(
            y_true, y_pred,
            labels=unique_labels,
            average='macro',
            zero_division=0
        )
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # 클래스별 정확도
        per_class_acc = self._calculate_per_class_accuracy(
            y_true, y_pred, unique_labels
        )
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc,
            'predictions': predictions,
            'labels': unique_labels
        }
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _extract_predicted_emotion(self, result) -> str:
        """
        분석 결과에서 예측 감정 추출
        """
        from bidirectional_propagation import Layer
        
        # DISCOURSE 레벨의 감정 사용
        emotion_vector = result.layer_emotions[Layer.DISCOURSE]
        emotion_dict = emotion_vector.to_dict()
        
        # 중립 제외
        emotion_dict.pop('neutral', None)
        
        # 가장 강한 감정
        if not emotion_dict:
            return 'neutral'
        
        predicted = max(emotion_dict.items(), key=lambda x: x[1])
        return predicted[0]
    
    def _calculate_per_class_accuracy(
        self,
        y_true: List[str],
        y_pred: List[str],
        labels: List[str]
    ) -> Dict:
        """
        클래스별 정확도
        """
        per_class = {}
        
        for label in labels:
            # 해당 클래스의 인덱스
            indices = [i for i, y in enumerate(y_true) if y == label]
            
            if not indices:
                per_class[label] = 0.0
                continue
            
            # 정확도 계산
            correct = sum(
                1 for i in indices if y_pred[i] == label
            )
            per_class[label] = correct / len(indices)
        
        return per_class
    
    def _print_results(self, results: Dict):
        """
        결과 출력
        """
        print("\n" + "="*70)
        print("평가 결과")
        print("="*70)
        
        print(f"\n전체 정확도: {results['accuracy']:.2%}")
        print(f"F1 Score:    {results['f1_score']:.2%}")
        
        print("\n클래스별 정확도:")
        print("-" * 70)
        for label, acc in sorted(
            results['per_class_accuracy'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar = "█" * int(acc * 30)
            print(f"  {label:15} {acc:6.2%} {bar}")
        
        # 혼동 행렬 시각화
        self.plot_confusion_matrix(
            results['confusion_matrix'],
            results['labels']
        )
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        save_path: str = "confusion_matrix.png"
    ):
        """
        혼동 행렬 시각화
        """
        plt.figure(figsize=(12, 10))
        
        # 정규화
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix (Normalized)', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 혼동 행렬 저장: {save_path}")


# ============================================================================
# 4. Before/After 비교
# ============================================================================

class BeforeAfterComparison:
    """
    기존 시스템 vs 새 시스템 비교
    """
    
    def __init__(self):
        self.results = {
            'before': None,
            'after': None
        }
    
    def compare(
        self,
        before_engine,
        after_engine,
        test_data: List[Dict]
    ):
        """
        두 시스템 비교
        """
        print("\n" + "="*70)
        print("Before/After 비교")
        print("="*70)
        
        evaluator = PerformanceEvaluator({})
        
        # Before (기존 HIT 시스템)
        print("\n[Before] 기존 HIT 시스템 평가...")
        self.results['before'] = evaluator.evaluate(
            before_engine, test_data, verbose=False
        )
        
        # After (완전 통합 시스템)
        print("\n[After] COSMOS v2.0 평가...")
        self.results['after'] = evaluator.evaluate(
            after_engine, test_data, verbose=False
        )
        
        # 비교 출력
        self._print_comparison()
        
        # 시각화
        self._plot_comparison()
    
    def _print_comparison(self):
        """
        비교 결과 출력
        """
        before = self.results['before']
        after = self.results['after']
        
        print("\n" + "="*70)
        print("비교 결과")
        print("="*70)
        
        # 정확도
        acc_before = before['accuracy']
        acc_after = after['accuracy']
        acc_improvement = ((acc_after - acc_before) / acc_before) * 100
        
        print(f"\n정확도:")
        print(f"  Before: {acc_before:.2%}")
        print(f"  After:  {acc_after:.2%}")
        print(f"  향상:   {acc_improvement:+.1f}%")
        
        # F1 Score
        f1_before = before['f1_score']
        f1_after = after['f1_score']
        f1_improvement = ((f1_after - f1_before) / f1_before) * 100
        
        print(f"\nF1 Score:")
        print(f"  Before: {f1_before:.2%}")
        print(f"  After:  {f1_after:.2%}")
        print(f"  향상:   {f1_improvement:+.1f}%")
        
        # 클래스별 비교
        print("\n클래스별 정확도 향상:")
        print("-" * 70)
        
        per_class_before = before['per_class_accuracy']
        per_class_after = after['per_class_accuracy']
        
        improvements = []
        for label in per_class_before.keys():
            if label in per_class_after:
                before_val = per_class_before[label]
                after_val = per_class_after[label]
                
                if before_val > 0:
                    improvement = ((after_val - before_val) / before_val) * 100
                else:
                    improvement = 0.0
                
                improvements.append((label, improvement, after_val))
        
        # 향상률 순으로 정렬
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        for label, improvement, after_val in improvements[:10]:
            print(f"  {label:15} {improvement:+6.1f}%  "
                  f"(→ {after_val:.2%})")
    
    def _plot_comparison(self, save_path: str = "before_after.png"):
        """
        비교 시각화
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        before = self.results['before']
        after = self.results['after']
        
        # 1. 전체 지표 비교
        ax = axes[0]
        
        metrics = ['Accuracy', 'F1 Score']
        before_values = [
            before['accuracy'] * 100,
            before['f1_score'] * 100
        ]
        after_values = [
            after['accuracy'] * 100,
            after['f1_score'] * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_values, width, 
                      label='Before', color='coral')
        bars2 = ax.bar(x + width/2, after_values, width, 
                      label='After', color='lightgreen')
        
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Overall Performance Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10)
        
        # 2. 클래스별 향상률
        ax = axes[1]
        
        per_class_before = before['per_class_accuracy']
        per_class_after = after['per_class_accuracy']
        
        improvements = {}
        for label in per_class_before.keys():
            if label in per_class_after:
                before_val = per_class_before[label]
                after_val = per_class_after[label]
                
                if before_val > 0:
                    improvement = ((after_val - before_val) / before_val) * 100
                    improvements[label] = improvement
        
        # 상위 10개
        top_improvements = sorted(
            improvements.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        labels = [label for label, _ in top_improvements]
        values = [value for _, value in top_improvements]
        
        colors = ['green' if v > 0 else 'red' for v in values]
        
        ax.barh(labels, values, color=colors, alpha=0.7)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Improvement (%)', fontsize=12)
        ax.set_title('Per-Class Accuracy Improvement (Top 10)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 비교 그래프 저장: {save_path}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """
    데이터셋 통합 + 평가 메인
    """
    print("="*70)
    print("COSMOS EMOTION - 데이터셋 통합 + 성능 평가")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n[1/5] 데이터셋 로드")
    print("-" * 70)
    
    loader = DatasetLoader(data_dir="./data")
    samples = loader.load_dataset()
    
    # 2. 전처리
    print("\n[2/5] 데이터 전처리")
    print("-" * 70)
    
    preprocessor = DataPreprocessor()
    train_data, val_data, test_data = preprocessor.preprocess(samples)
    
    # 3. 엔진 로드
    print("\n[3/5] 엔진 초기화")
    print("-" * 70)
    
    from integrated_cosmos_system import IntegratedCOSMOSEngine
    
    # Before (기존 HIT만)
    # 실제로는 이전 버전의 엔진을 사용
    before_engine = IntegratedCOSMOSEngine(
        use_konlpy=False,
        propagation_iterations=0  # 전파 없음
    )
    
    # After (완전 통합)
    after_engine = IntegratedCOSMOSEngine(
        use_konlpy=False,
        propagation_iterations=2
    )
    
    print("✓ 엔진 준비 완료")
    
    # 4. 평가
    print("\n[4/5] 성능 평가")
    print("-" * 70)
    
    evaluator = PerformanceEvaluator(preprocessor.label_decoder)
    
    # 테스트 데이터로 평가 (샘플만)
    test_sample = test_data[:100]  # 처음 100개만
    
    results = evaluator.evaluate(after_engine, test_sample)
    
    # 5. Before/After 비교
    print("\n[5/5] Before/After 비교")
    print("-" * 70)
    
    comparator = BeforeAfterComparison()
    comparator.compare(before_engine, after_engine, test_sample)
    
    print("\n" + "="*70)
    print("완료!")
    print("="*70)


if __name__ == "__main__":
    main()
