# 프로젝트 파일 목록 및 설명

COSMOS_EMOTION 프로젝트의 모든 파일에 대한 상세 설명입니다.

## 📄 문서 파일

| 파일명 | 설명 | 최종 수정일 |
|--------|------|------------|
| readme.md | 프로젝트 메인 문서 (한국어) - 시스템 개요, 설치 방법, 사용 예제 | 2025-01-06 |
| EN_README.MD | 프로젝트 메인 문서 (영어) - 영문 버전 문서 | 2025-01-06 |
| MARKDOWN_GUIDE.md | 마크다운 사용 가이드 - 이미지/링크 삽입 방법 설명 | 2025-01-06 |
| LICENSE | Apache License 2.0 라이선스 파일 | 2025-01-06 |
| 설치 및 배포 파일.txt | 설치 및 배포 관련 설정 파일 모음 (requirements, Dockerfile, 등) | 2025-01-06 |

## 🐍 Python 소스 파일

| 파일명 | 설명 | 최종 수정일 |
|--------|------|------------|
| integrated_cosmos_system.py | 통합 COSMOS 감정 분석 엔진 - 전체 시스템의 메인 통합 모듈 | 2025-01-06 |
| morpheme_intensity_system.py | 형태소 강도 분석 시스템 - 형태소 분석 및 조사/어미 강도 계산 | 2025-01-06 |
| bidirectional_propagation.py | 양방향 계층 전파 시스템 - 5개 계층 간 양방향 감정 전파 처리 | 2025-01-06 |
| resonance_system.py | 다중 채널 공명 시스템 - 5채널 공명 감지 (Spectral, Phase, Harmonic, Semantic, Cross-Layer) | 2025-01-06 |
| api_server.py | FastAPI 기반 REST API 서버 - 웹 API 엔드포인트 제공 | 2025-01-06 |
| dataset_integration.py | 데이터셋 통합 및 성능 평가 - AIHub 데이터셋 로드 및 전처리, 모델 평가 | 2025-01-06 |
| visualization_comparison.py | 시각화 및 성능 비교 - ES Timeline 시각화, 계층별 감정 흐름 그래프 생성 | 2025-01-06 |
| quick_start_example.py | 빠른 시작 예제 - 전체 시스템 테스트 및 데모 실행 스크립트 | 2025-01-06 |

## ⚙️ 설정 파일

| 파일명 | 설명 | 최종 수정일 |
|--------|------|------------|
| requirements.txt | Python 패키지 의존성 목록 - 필요한 라이브러리 명시 | 2025-01-06 |
| .gitignore | Git 제외 파일 목록 - 버전 관리에서 제외할 파일 패턴 | 2025-01-06 |

## 📁 폴더 구조

```
COSMOS_EMOTION/
├── images/                    # 이미지 및 다이어그램 저장 폴더
│   ├── architecture/          # 시스템 아키텍처 이미지
│   ├── diagrams/             # 플로우 다이어그램
│   └── screenshots/          # 스크린샷 및 결과 이미지
├── readme.md                 # 메인 문서 (한국어)
├── EN_README.MD              # 메인 문서 (영어)
├── MARKDOWN_GUIDE.md         # 마크다운 가이드
├── integrated_cosmos_system.py    # 통합 시스템
├── morpheme_intensity_system.py   # 형태소 분석
├── bidirectional_propagation.py   # 양방향 전파
├── resonance_system.py            # 공명 시스템
├── api_server.py                  # API 서버
├── dataset_integration.py         # 데이터셋 통합
├── visualization_comparison.py    # 시각화
├── quick_start_example.py         # 빠른 시작
└── requirements.txt               # 의존성

## 🔄 파일 간 의존성

```
quick_start_example.py (데모 실행)
    └── integrated_cosmos_system.py (메인 엔진)
        ├── morpheme_intensity_system.py (형태소 분석)
        ├── bidirectional_propagation.py (계층 전파)
        └── resonance_system.py (공명 시스템)

api_server.py (API 서버)
    └── integrated_cosmos_system.py (메인 엔진)

dataset_integration.py (데이터셋 평가)
    └── integrated_cosmos_system.py (메인 엔진)

visualization_comparison.py (시각화)
    └── integrated_cosmos_system.py (메인 엔진)
```

## 🚀 주요 파일 사용 방법

### 1. 빠른 시작
```bash
python quick_start_example.py
```

### 2. API 서버 실행
```bash
python api_server.py
```

### 3. 데이터셋 평가
```bash
python dataset_integration.py
```

### 4. 시각화 생성
```bash
python visualization_comparison.py
```

## 📝 파일 수정 이력

| 날짜 | 파일 | 변경 내용 |
|------|------|----------|
| 2025-01-06 | FILE_DESCRIPTIONS.md | 파일 설명 문서 최초 생성 |
| 2025-01-06 | MARKDOWN_GUIDE.md | 마크다운 가이드 추가 |
| 2025-01-06 | images/ | 이미지 폴더 생성 |

## 💡 참고 사항

- 모든 Python 파일은 UTF-8 인코딩을 사용합니다
- 한글 주석과 docstring이 포함되어 있습니다
- Python 3.8 이상 버전이 필요합니다
- 자세한 설치 방법은 `readme.md`를 참조하세요

---

문서 작성일: 2025-01-06  
최종 수정일: 2025-01-06
