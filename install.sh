#!/bin/bash
# COSMOS EMOTION v2.0 설치 스크립트

echo "========================================"
echo "COSMOS EMOTION v2.0 설치"
echo "========================================"
echo ""

# Python 버전 확인
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 버전: $python_version"

if [[ $(echo $python_version | cut -d. -f1,2) < "3.8" ]]; then
    echo "⚠️  Python 3.8 이상이 필요합니다."
    exit 1
fi

# 가상환경 생성
echo ""
echo "[1/5] 가상환경 생성..."
python3 -m venv venv

# 가상환경 활성화
echo "[2/5] 가상환경 활성화..."
source venv/bin/activate

# 패키지 업그레이드
echo "[3/5] pip 업그레이드..."
pip install --upgrade pip

# 패키지 설치
echo "[4/5] 패키지 설치..."
pip install -r requirements.txt

# 선택적 패키지 설치
echo ""
read -p "KoNLPy를 설치하시겠습니까? (더 높은 정확도) [y/N]: " install_konlpy
if [[ $install_konlpy =~ ^[Yy]$ ]]; then
    echo "KoNLPy 설치 중..."
    pip install konlpy
    
    # macOS의 경우
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS 감지 - Java 설치 확인..."
        if ! command -v java &> /dev/null; then
            echo "⚠️  Java가 필요합니다. Homebrew로 설치:"
            echo "    brew install openjdk@11"
        fi
    fi
fi

# 테스트
echo ""
echo "[5/5] 설치 확인..."
python3 -c "
try:
    import numpy
    import matplotlib
    import scipy
    print('✓ 기본 패키지 로드 성공')
except Exception as e:
    print(f'⚠️  패키지 로드 실패: {e}')
    exit(1)
"

# 완료
echo ""
echo "========================================"
echo "✓ 설치 완료!"
echo "========================================"
echo ""
echo "실행 방법:"
echo "  1. 가상환경 활성화:"
echo "     source venv/bin/activate"
echo ""
echo "  2. 메인 프로그램 실행:"
echo "     python quick_start_example.py"
echo ""
echo "  3. API 서버 실행:"
echo "     python api_server.py"
echo "     또는: uvicorn api_server:app --reload"
echo ""
echo "  4. Makefile 도움말:"
echo "     make help"
echo ""
