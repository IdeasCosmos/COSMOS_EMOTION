@echo off
REM COSMOS EMOTION v2.0 설치 스크립트 (Windows)

echo ========================================
echo COSMOS EMOTION v2.0 설치
echo ========================================
echo.

REM Python 버전 확인
python --version
if %errorlevel% neq 0 (
    echo ⚠️  Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 다운로드
    pause
    exit /b 1
)

REM 가상환경 생성
echo.
echo [1/5] 가상환경 생성...
python -m venv venv

REM 가상환경 활성화
echo [2/5] 가상환경 활성화...
call venv\Scripts\activate.bat

REM 패키지 업그레이드
echo [3/5] pip 업그레이드...
python -m pip install --upgrade pip

REM 패키지 설치
echo [4/5] 패키지 설치...
pip install -r requirements.txt

REM 테스트
echo.
echo [5/5] 설치 확인...
python -c "import numpy; import matplotlib; import scipy; print('✓ 설치 성공')"

REM 완료
echo.
echo ========================================
echo ✓ 설치 완료!
echo ========================================
echo.
echo 실행 방법:
echo   1. 가상환경 활성화:
echo      venv\Scripts\activate.bat
echo.
echo   2. 메인 프로그램 실행:
echo      python quick_start_example.py
echo.
echo   3. API 서버 실행:
echo      python api_server.py
echo.
pause
