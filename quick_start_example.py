"""
COSMOS EMOTION v2.0 - 빠른 시작 예제
=====================================

이 파일을 실행하면 전체 시스템을 바로 테스트할 수 있습니다.

실행 방법:
    python main.py

또는 특정 문장 분석:
    python main.py "분석할 문장을 여기에 입력"
"""

import sys
import time
import numpy as np


# ============================================================================
# 설정
# ============================================================================

# 테스트 문장들 (다양한 감정 패턴)
TEST_SENTENCES = [
    # 1. 단순 긍정
    "정말 기쁘네요! 오늘은 최고의 날이에요.",
    
    # 2. 역접 (지만) - 감정 반전
    "슬프지만 견뎌야 해.",
    
    # 3. 인터넷 표현
    "너무 짜증나ㅋㅋㅋ",
    
    # 4. 복합 감정 (회상 + 슬픔)
    "오래된 앨범 속에서 환하게 웃고 있는 친구의 모습을 보니 "
    "반가웠지만, 이제는 다시 볼 수 없다는 생각에 가슴 한편이 아려왔다.",
    
    # 5. 강조 조사 (조차, 까지)
    "친구조차 날 이해하지 못해. 끝까지 노력했는데도.",
    
    # 6. 의문문 (혼란)
    "이게 정말 맞는 걸까? 도대체 뭐가 문제인지 모르겠어.",
    
    # 7. 다중 감정 중첩
    "엄마가 보고 싶어. 화가 나면서도 슬프고, 그리우면서도 미안해.",
]


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """
    메인 실행 함수
    """
    print("="*80)
    print("COSMOS EMOTION v2.0 - 완전 통합 시스템")
    print("="*80)
    print()
    
    # ========================================================================
    # 1. 시스템 초기화
    # ========================================================================
    print("[1/4] 시스템 초기화 중...")
    print("-" * 80)
    
    try:
        from integrated_cosmos_system import IntegratedCOSMOSEngine
        
        engine = IntegratedCOSMOSEngine(
            use_konlpy=False,  # KoNLPy 없으면 자체 파서
            fps=25,
            propagation_iterations=2
        )
        
        print("✓ 형태소 분석기 초기화")
        print("✓ 양방향 전파 엔진 로드")
        print("✓ 5채널 공명 시스템 준비")
        print("✓ 시각화 도구 로드")
        print()
        
    except ImportError as e:
        print(f"⚠ 모듈 로드 실패: {e}")
        print("\n필수 파일을 확인하세요:")
        print("  - morpheme_intensity_system.py")
        print("  - bidirectional_propagation.py")
        print("  - resonance_system.py")
        print("  - integrated_cosmos_system.py")
        return
    
    # ========================================================================
    # 2. 분석할 텍스트 선택
    # ========================================================================
    print("[2/4] 분석 텍스트 선택")
    print("-" * 80)
    
    # 커맨드 라인 인자 확인
    if len(sys.argv) > 1:
        # 사용자 입력 텍스트
        text_to_analyze = " ".join(sys.argv[1:])
        print(f"사용자 입력: {text_to_analyze}")
    else:
        # 테스트 문장 선택
        print("테스트 문장 목록:")
        for i, sentence in enumerate(TEST_SENTENCES, 1):
            preview = sentence[:50] + "..." if len(sentence) > 50 else sentence
            print(f"  {i}. {preview}")
        
        print()
        choice = input("분석할 문장 번호 (1-7, 엔터=4번): ").strip()
        
        if not choice:
            choice = "4"
        
        try:
            idx = int(choice) - 1
            text_to_analyze = TEST_SENTENCES[idx]
        except (ValueError, IndexError):
            print("⚠ 잘못된 입력. 기본 문장(4번) 사용.")
            text_to_analyze = TEST_SENTENCES[3]
    
    print()
    print("선택된 텍스트:")
    print(f"  📝 {text_to_analyze}")
    print()
    
    # ========================================================================
    # 3. 감정 분석 실행
    # ========================================================================
    print("[3/4] 감정 분석 실행")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        result = engine.analyze(text_to_analyze)
        
    except Exception as e:
        print(f"⚠ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return
    
    elapsed = (time.time() - start_time) * 1000
    
    print()
    print(f"✓ 분석 완료! (소요 시간: {elapsed:.1f}ms)")
    print()
    
    # ========================================================================
    # 4. 결과 출력
    # ========================================================================
    print("[4/4] 결과 출력")
    print("-" * 80)
    
    # 상세 결과 출력
    engine.print_result(result)
    
    # ========================================================================
    # 5. 시각화 (옵션)
    # ========================================================================
    print()
    print("="*80)
    print("시각화 생성")
    print("="*80)
    
    visualize = input("시각화 파일을 생성하시겠습니까? (y/n, 기본=y): ").strip().lower()
    
    if not visualize or visualize == 'y':
        try:
            from visualization_comparison import visualize_all
            
            print("\n시각화 생성 중...")
            visualize_all(result)
            
            print("\n✓ 시각화 완료!")
            print("\n생성된 파일:")
            print("  📊 timeline.png - ES Timeline (악보)")
            print("  📊 layer_flow.png - 계층별 감정 흐름")
            print("  📊 resonance.png - 공명 패턴")
            print("  📊 comparison.png - 성능 비교")
            
        except Exception as e:
            print(f"\n⚠ 시각화 생성 실패: {e}")
            print("  matplotlib이 설치되어 있는지 확인하세요:")
            print("  pip install matplotlib")
    
    # ========================================================================
    # 추가 분석 옵션
    # ========================================================================
    print()
    print("="*80)
    print("추가 옵션")
    print("="*80)
    print()
    print("1. 다른 문장 분석")
    print("2. 배치 분석 (전체 테스트 세트)")
    print("3. 벤치마크 실행")
    print("4. 종료")
    print()
    
    option = input("선택 (1-4, 기본=4): ").strip()
    
    if option == "1":
        print("\n프로그램을 다시 실행하거나 직접 텍스트를 입력하세요:")
        print("  python main.py \"분석할 문장\"")
    
    elif option == "2":
        print("\n배치 분석 시작...")
        batch_analyze(engine, TEST_SENTENCES)
    
    elif option == "3":
        print("\n벤치마크 실행...")
        run_benchmark(engine)
    
    print()
    print("="*80)
    print("COSMOS EMOTION - 감사합니다!")
    print("="*80)


# ============================================================================
# 추가 기능
# ============================================================================

def batch_analyze(engine, sentences):
    """
    배치 분석
    """
    print("-" * 80)
    results = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n[{i}/{len(sentences)}] {sentence[:40]}...")
        
        start = time.time()
        result = engine.analyze(sentence)
        elapsed = (time.time() - start) * 1000
        
        results.append({
            'text': sentence,
            'result': result,
            'time': elapsed
        })
        
        # 간단한 요약
        discourse_emotion = result.layer_emotions[
            engine.Layer.DISCOURSE
        ].to_dict()
        
        top_emotions = sorted(
            discourse_emotion.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        print(f"  주요 감정: {top_emotions[0][0]} ({top_emotions[0][1]:.2f})")
        print(f"  처리 시간: {elapsed:.1f}ms")
    
    # 통계
    print("\n" + "="*80)
    print("배치 분석 통계")
    print("="*80)
    
    avg_time = np.mean([r['time'] for r in results])
    avg_amp = np.mean([
        r['result'].amplification['total_amplification'] 
        for r in results
    ])
    
    print(f"평균 처리 시간: {avg_time:.1f}ms")
    print(f"평균 증폭률: ×{avg_amp:.2f}")
    
    # 공명 패턴 통계
    total_patterns = sum(
        sum(len(p) for p in r['result'].resonance_patterns.values())
        for r in results
    )
    print(f"총 공명 패턴: {total_patterns}개")


def run_benchmark(engine):
    """
    벤치마크
    """
    print("-" * 80)
    
    # 다양한 길이의 텍스트
    test_cases = [
        ("짧은 문장", "기쁘다"),
        ("중간 문장", "정말 기쁘네요! 오늘은 최고의 날이에요."),
        ("긴 문장", TEST_SENTENCES[3]),
        ("매우 긴 문장", " ".join(TEST_SENTENCES[:3]))
    ]
    
    print("\n벤치마크 실행 중...\n")
    
    for name, text in test_cases:
        times = []
        
        # 10회 반복
        for _ in range(10):
            start = time.time()
            engine.analyze(text)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"{name:15} ({len(text):3}자): "
              f"{avg_time:6.1f}ms ± {std_time:5.1f}ms")
    
    print("\n✓ 벤치마크 완료!")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n⚠ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
