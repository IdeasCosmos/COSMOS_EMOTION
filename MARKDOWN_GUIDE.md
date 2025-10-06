# Markdown 사용 가이드

이 문서는 README 파일에서 이미지와 링크를 사용하는 방법을 설명합니다.

## 📸 이미지 삽입 방법

### 기본 문법
```markdown
![대체 텍스트](이미지 경로)
```

### 예시

#### 1. 로컬 이미지 사용 (저장소 내 이미지)
```markdown
![시스템 아키텍처](images/system-architecture.png)
![감정 흐름도](images/emotion-flow.png)
```

#### 2. 절대 경로 사용 (GitHub 저장소)
```markdown
![시스템 다이어그램](https://raw.githubusercontent.com/IdeasCosmos/COSMOS_EMOTION/main/images/diagram.png)
```

#### 3. 이미지에 링크 추가
```markdown
[![클릭 가능한 이미지](images/demo.png)](https://example.com)
```

#### 4. 이미지 크기 조절 (HTML 사용)
```markdown
<img src="images/logo.png" alt="로고" width="200"/>
```

## 🔗 링크 생성 방법

### 기본 문법
```markdown
[링크 텍스트](URL)
```

### 예시

#### 1. 외부 링크
```markdown
[공식 문서](https://docs.example.com)
[GitHub 저장소](https://github.com/IdeasCosmos/COSMOS_EMOTION)
```

#### 2. 내부 링크 (같은 문서 내)
```markdown
[설치 방법으로 이동](#설치)
[빠른 시작](#빠른-시작)
```

#### 3. 다른 파일 링크
```markdown
[기여 가이드 참조](CONTRIBUTING.md)
[라이선스 확인](LICENSE)
```

#### 4. 이메일 링크
```markdown
[이메일 보내기](mailto:sjpupro@gmail.com)
```

#### 5. 참조 스타일 링크
```markdown
자세한 내용은 [문서][1]를 참조하세요.

[1]: https://docs.example.com
```

## 📁 이미지 폴더 구조

### 권장 폴더 구조
```
COSMOS_EMOTION/
├── images/               # 이미지 저장 폴더
│   ├── architecture/     # 아키텍처 관련 이미지
│   ├── diagrams/         # 다이어그램
│   ├── screenshots/      # 스크린샷
│   └── logos/            # 로고 및 아이콘
├── readme.md
└── EN_README.MD
```

### 이미지 파일 명명 규칙
- 소문자와 하이픈 사용: `emotion-flow-diagram.png`
- 명확한 이름 사용: `system-architecture.png` ✅ (좋음)
- 일반적인 이름 피하기: `image1.png` ❌ (나쁨)

## 🎯 실전 예제

### README에 이미지 섹션 추가

```markdown
## 시스템 아키텍처

전체 시스템의 구조는 다음과 같습니다:

![시스템 아키텍처 다이어그램](images/architecture/system-overview.png)

### 감정 상태 전이 흐름

![감정 전이 플로우](images/diagrams/emotion-transition-flow.png)

자세한 내용은 [기술 문서](docs/technical-specs.md)를 참조하세요.
```

## 💡 팁과 모범 사례

### 1. 이미지 최적화
- 파일 크기를 적절히 유지 (보통 < 1MB)
- PNG: 다이어그램, 로고, 스크린샷
- JPG: 사진
- SVG: 벡터 그래픽 (가능한 경우)

### 2. 대체 텍스트 작성
```markdown
![감정 분석 결과 - 기쁨 60%, 슬픔 20%, 놀람 20%](images/analysis-result.png)
```
- 이미지가 로드되지 않을 때 표시됨
- 접근성 향상 (스크린 리더 사용자)

### 3. 상대 경로 vs 절대 경로
- **상대 경로**: 저장소 내부 파일 참조 시 사용
  ```markdown
  ![로고](images/logo.png)
  ```
- **절대 경로**: 안정적인 외부 호스팅 필요 시
  ```markdown
  ![로고](https://raw.githubusercontent.com/user/repo/main/images/logo.png)
  ```

### 4. .gitignore 설정
이미지가 너무 크거나 임시 파일인 경우:
```gitignore
# 이미지 중 임시 파일만 제외
images/temp/
*.tmp.png
```

## 📋 체크리스트

이미지를 추가할 때:
- [ ] 이미지가 `images/` 폴더에 있는가?
- [ ] 파일명이 명확하고 일관적인가?
- [ ] 대체 텍스트를 작성했는가?
- [ ] 이미지 파일 크기가 적절한가?
- [ ] 커밋 전에 이미지가 제대로 표시되는지 확인했는가?

## 🔍 GitHub에서 미리보기

GitHub에서 이미지가 제대로 표시되는지 확인하려면:

1. 변경사항 커밋 및 푸시
2. GitHub 저장소 페이지에서 README 확인
3. 이미지가 깨진 경우 경로 재확인

---

이 가이드가 도움이 되었기를 바랍니다! 질문이 있으면 [이슈](https://github.com/IdeasCosmos/COSMOS_EMOTION/issues)를 열어주세요.
