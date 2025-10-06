.PHONY: help install test run api docker clean

help:
	@echo "COSMOS EMOTION v2.0 - 명령어"
	@echo ""
	@echo "  make install    - 패키지 설치"
	@echo "  make test       - 테스트 실행"
	@echo "  make run        - 메인 프로그램 실행"
	@echo "  make api        - API 서버 실행"
	@echo "  make docker     - Docker 빌드 및 실행"
	@echo "  make clean      - 캐시 및 임시 파일 삭제"
	@echo ""

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov

run:
	python quick_start_example.py

api:
	uvicorn api_server:app --reload --port 8000

docker:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
