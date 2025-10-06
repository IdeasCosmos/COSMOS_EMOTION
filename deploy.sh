#!/bin/bash
# 프로덕션 배포 스크립트

echo "========================================"
echo "COSMOS EMOTION v2.0 프로덕션 배포"
echo "========================================"
echo ""

# 1. 코드 업데이트
echo "[1/6] Git Pull..."
git pull origin main

# 2. 의존성 업데이트
echo "[2/6] 패키지 업데이트..."
pip install -r requirements.txt --upgrade

# 3. 테스트
echo "[3/6] 테스트 실행..."
if [ -d "tests" ]; then
    pytest tests/ -v
    if [ $? -ne 0 ]; then
        echo "⚠️  테스트 실패! 배포 중단."
        exit 1
    fi
else
    echo "테스트 디렉토리가 없습니다. 테스트 단계를 건너뜁니다."
fi

# 4. Docker 이미지 빌드
echo "[4/6] Docker 이미지 빌드..."
docker-compose build

# 5. 기존 컨테이너 중지
echo "[5/6] 기존 컨테이너 중지..."
docker-compose down

# 6. 새 컨테이너 시작
echo "[6/6] 새 컨테이너 시작..."
docker-compose up -d

# 헬스체크
echo ""
echo "헬스체크 대기..."
sleep 10

curl -f http://localhost:8000/health
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 배포 성공!"
    echo "API: http://localhost:8000/docs"
else
    echo ""
    echo "⚠️  헬스체크 실패!"
    docker-compose logs --tail=50
    exit 1
fi
