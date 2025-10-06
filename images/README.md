# Images 폴더

이 폴더는 COSMOS_EMOTION 프로젝트의 모든 이미지 파일을 저장합니다.

## 📁 폴더 구조

```
images/
├── architecture/       # 시스템 아키텍처 다이어그램
├── diagrams/          # 플로우차트 및 다이어그램
├── screenshots/       # 스크린샷 및 실행 결과
└── logos/            # 로고 및 아이콘
```

## 📝 사용 방법

### README에서 이미지 참조

```markdown
![시스템 아키텍처](images/architecture/system-overview.png)
![감정 흐름도](images/diagrams/emotion-flow.png)
```

## 🎨 이미지 가이드라인

### 파일 명명 규칙
- 소문자와 하이픈 사용
- 명확하고 설명적인 이름
- 예: `emotion-transition-flow.png`, `7-layer-architecture.png`

### 파일 형식
- **PNG**: 다이어그램, 로고, 투명 배경 필요 시
- **JPG**: 사진, 스크린샷
- **SVG**: 벡터 그래픽 (확대/축소 시 품질 유지)

### 파일 크기
- 일반 이미지: 최대 500KB
- 상세 다이어그램: 최대 1MB
- 압축 도구를 사용하여 최적화

## 📋 예정된 이미지 목록

README에서 참조되는 이미지들:

### Architecture
- [ ] `architecture/system-overview.png` - 전체 시스템 개요
- [ ] `architecture/7-layer-model.png` - 7계층 모델 구조
- [ ] `architecture/duality-architecture.png` - 이중성 아키텍처

### Diagrams
- [ ] `diagrams/emotion-transition-flow.png` - 감정 상태 전이 흐름
- [ ] `diagrams/multi-vector-space.png` - 다중 벡터 감정 공간
- [ ] `diagrams/cascade-control.png` - 캐스케이드 제어 프로세스

### Screenshots
- [ ] `screenshots/demo-output.png` - 데모 실행 결과
- [ ] `screenshots/api-docs.png` - API 문서 화면
- [ ] `screenshots/visualization.png` - 시각화 결과

### Performance
- [ ] `performance/benchmark-chart.png` - 성능 벤치마크 그래프
- [ ] `performance/comparison-table.png` - BERT 비교 결과

## 🔧 이미지 생성 도구

### 다이어그램 생성
- [Draw.io](https://app.diagrams.net/) - 플로우차트
- [Excalidraw](https://excalidraw.com/) - 손그림 스타일 다이어그램
- [PlantUML](https://plantuml.com/) - 텍스트 기반 UML

### 최적화 도구
- [TinyPNG](https://tinypng.com/) - PNG 압축
- [ImageOptim](https://imageoptim.com/) - 이미지 최적화

## 📌 참고사항

1. **Git LFS** (Large File Storage)가 필요한 경우:
   - 매우 큰 이미지 파일 (>10MB)
   - 비디오 파일
   
2. **저작권**: 
   - 직접 생성한 이미지만 업로드
   - 외부 이미지 사용 시 라이선스 확인

3. **대체 텍스트**:
   - 모든 이미지에 설명적인 alt 텍스트 제공
   - 접근성 향상

---

자세한 마크다운 사용법은 [MARKDOWN_GUIDE.md](../MARKDOWN_GUIDE.md)를 참조하세요.
