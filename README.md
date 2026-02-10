# 🏠 AIVLE 스쿨 빅 프로젝트 16조

> **Public** / 🏠 **RAG도 락이다!!**

# 🏢 AI 기반 입찰 인텔리전스 플랫폼

<!-- 프로젝트 대표 이미지 추가해야함 -->
<div align="center">
  <img src="./ops/images/Screenshot/썸네일.jpg" alt="AI 기반 입찰 인텔리전스 플랫폼 메인 화면" width="800"/>
  
  <!-- 배지들, 최종이후에 올릴예정-->
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
  [![React](https://img.shields.io/badge/react-18.0+-blue.svg)](https://reactjs.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
  
<!-- BidAssitance wiki 페이지 만들어서 링크 수정할 것 -->
**📖 [Documentation Wiki](https://github.com/LxNx-Hn/chatbot-with-kt-dgucenter/wiki/%EC%9C%84%ED%82%A4%E2%80%90%EB%AC%B8%EC%84%9C%EB%AA%A8%EC%9D%8C)**
</div>

---

## 📋 목차

1. [👥 팀 구성](#-팀-구성)
2. [📁 리포지토리 구조](#-리포지토리-구조)
3. [📖 프로젝트 개요](#-프로젝트-개요)
4. [💡 기능 소개](#-기능-소개)
5. [▶️ 시연 영상](#️-시연-영상)
6. [🏗️ 구현 상세](#️-구현-상세)
7. [🛠️ 배포 가이드](#-배포-가이드)
8. [👨‍💻 팀원별 상세 업무](#-팀원별-상세-업무)
9. [📝 회의록 및 개발 과정](#-회의록-및-개발-과정)
10. [💬 팀원 한마디](#-팀원-한마디)

---

## 👥 팀 구성

<div align="center">
  <table>
    <tr>
      <td align="center" width="200px">
        <img src="./ops/images/team/전보윤.jpg" width="120px" alt="전보윤"/><br>
        <b>전보윤</b><br>
        <sub>PM, AI</sub><br>
        <a href="https://github.com/sprintmonster">@sprintmonster</a>
      </td>
      <td align="center" width="200px">
        <img src="./ops/images/team/강버들.jpg" width="120px" alt="강버들"/><br>
        <b>강버들</b><br>
        <sub>FE</sub><br>
        <a href="https://github.com/LxNx-Hn">@LxNx-Hn</a>
      </td>
      <td align="center" width="200px">
        <img src="./ops/images/team/김형모.jpg" width="120px" alt="김형모"/><br>
        <b>김형모</b><br>
        <sub>BE</sub><br>
        <a href="https://github.com/HyeongmoKim">@HyeongmoKim</a>
      </td>
      <td align="center" width="200px">
        <img src="./ops/images/team/공동욱.jpg" width="120px" alt="공동욱"/><br>
        <b>공동욱</b><br>
        <sub>BE</sub><br>
        <a href="https://github.com/cos65536">@cos65536</a>
      </td>
    </tr>
  </table>
  <table>
    <tr>
      <td align="center" width="200px">
        <img src="./ops/images/team/윤성빈.jpg" width="120px" alt="윤성빈"/><br>
        <b>윤성빈</b><br>
        <sub>AI</sub><br>
        <a href="https://github.com/ysb1208">@ysb1208</a>
      </td>
      <td align="center" width="200px">
        <img src="./ops/images/team/이세용.jpg" width="120px" alt="이세용"/><br>
        <b>이세용</b><br>
        <sub>BE</sub><br>
        <a href="https://github.com/pla2n">@pla2n</a>
      </td>
      <td align="center" width="200px">
        <img src="./ops/images/team/장우진.jpg" width="120px" alt="장우진"/><br>
        <b>장우진</b><br>
        <sub>AI</sub><br>
        <a href="https://github.com/jwj9456">@jwj9456</a>
      </td>
      <td align="center" width="200px">
        <img src="./ops/images/team/황도윤.jpg" width="120px" alt="황도윤"/><br>
        <b>황도윤</b><br>
        <sub>BE</sub><br>
        <a href="https://github.com/HDYee">@HDYee</a>
      </td>
    </tr>
  </table>
</div>

---

## 📁 리포지토리 구조

<!-- 프로젝트 최종 병합 후 수정할 것 -->

<details>
<summary><b>📂 상세 폴더 구조 보기</b></summary>

```
chatbot-with-kt-dgucenter/
├── .gitignore                  # Git 무시 파일
├── README.md                   # 프로젝트 설명서
├── docker-compose.yml          # 도커 컴포즈 설정
├── .github/                    # CI /CD 폴더
│   └──  workflows/             # Github 액션 파일
├──     └── deploy.yml          # CI/CD 자동화 문서
├── ops/                        # 문서 및 운영 자료
│   ├── images/                 # README용 이미지
│   ├── docs/                   # 추가 문서
│   └── presentations/          # 발표 자료
└── DSL_CHAT_BOT/               # 챗봇 소스코드
    ├── backend/
    │   ├── Dockerfile          # 백엔드 도커파일
    │   ├── config/
    │   │   ├── __init__.py
    │   │   ├── constants.py    # 카테고리 상수, 모델명
    │   │   └── settings.py     # API 키 불러오기, API URL, DB 경로
    │   ├── data/
    │   │   ├── final_data.csv  # 사업장 데이터
    │   │   └── master_summary_final.csv # 창업률 통계
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── embedding_model.py # ko-sroberta 임베딩
    │   │   └── llm_model.py    # Midm-2.0-Mini LLM
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── labeling.py     # 질문 카테고리 라벨링
    │   │   ├── policy_service.py # 정책 정보 검색
    │   │   ├── startup_service.py # 창업 데이터 분석
    │   │   └── trend_service.py # 트렌드 분석
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   └── text_processor.py # CSV→텍스트 변환, 동의어/유의어 사전
    │   ├── main.py             # FastAPI 서버
    │   ├── rag_llm.py          # 호환성 레이어
    │   ├── requirements.txt    # Python 의존성
    │   └── .env                # API키 저장 <- 로컬에서 사용시 생성필요
    └── frontend/
        ├── Dockerfile          # 프론트엔드 도커파일
        ├── netlify/                  # Netlify Functions 폴더
        │   └── functions/
        │       └── chat.js           # Functions 핸들러
        ├── public/
        ├── src/
        │   ├── components/
        │   │   ├── ChatBot.jsx       # 메인 챗봇 컴포넌트
        │   │   ├── ChatBubble.jsx    # 채팅 말풍선
        │   │   ├── ChatHeader.jsx    # 챗봇 헤더
        │   │   ├── ChatInput.jsx     # 입력창
        │   │   ├── ChatMessages.jsx  # 메시지 목록
        │   │   └── ThemeToggle.jsx   # 테마 토글
        │   ├── context/
        │   │   └── ThemeContext.jsx  # 테마 컨텍스트
        │   ├── styles/
        │   │   └── ChatStyles.css    # 스타일시트
        │   ├── App.jsx               # 메인 앱
        │   └── index.js              # 엔트리 포인트
        ├── netlify.toml              # Netlify 설정 파일
        ├── .env                      # 백엔드 요청경로 <- 로컬사용시 생성필요
        └── package.json              # Node.js 의존성
```

</details>

---

## 📖 프로젝트 소개

소형·중형 건설사를 위한 **AI 기반 입찰 인텔리전스 플랫폼​**입니다.

### 🎯 프로젝트 목표

AI를 활용한 지능형 검색, 공고 상세 분석을 통해서 누구든지 저비용으로 간편하게 정보를 얻는 입찰 플랫폼 구축​

#### 프로젝트 구동 화면

> **🏠 메인화면**
>
> <div align="center">
>     <img src="./ops/images/Screenshot/Light.png" alt="데스크탑 라이트" width="400"/>
>     <img src="./ops/images/Screenshot/Dark.png" alt="데스크탑 다크" width="400"/> 
> </div>
> <br>

> **💼 창업 질문 답변 예시**
>
> <div align="center">
>   <img src="./ops/images/Screenshot/예시_창업.png" alt="답변예시_창업" width="800"/>
> </div>
> <br>

> **📋 정책 질문 답변 예시**
>
> <div align="center">
>   <img src="./ops/images/Screenshot/예시_정책.png" alt="답변예시_정책" width="800"/>
> </div>
> <br>

> **📈 트렌드 질문 답변 예시**
>
> <div align="center">
>   <img src="./ops/images/Screenshot/예시_트렌드.png" alt="답변예시_트렌드" width="800"/>
> </div>
> <br>

> **📱 반응형 UI 설계**
>
> <div align="center">
>   <img src="./ops/images/Screenshot/반응형UI.png" alt="반응형" width="800"/>
> </div>
> <br>

---

## 💡 기능 소개

### 🎯 맞춤형 추천 대시보드

- 사용자의 프로필과 과거 조회 이력 분석하여, 수주 가능성 높은 공고를 추천하는 대시보드

### 🤖 RAG 기반 챗봇

- RAG 챗봇을 기반으로 원하는 공고를 자연어로 검색 가능

### 📄 공고문 요약 AI

- RAG가 적용된 기술이 첨부파일을 자동 분석하여 핵심 정보(자격, 과업, 독소 조항) 즉시 요약 제공

### 🔍 커뮤니티

- 질문, 정보, 후기, 토론 커뮤니티를 통해 입찰 정보 공유

---

## 🛠️ 기술 스택

<div align="center">

### Frontend

<img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black">
<img src="https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white">
<img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white">

### Backend

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/spring-%236DB33F.svg?style=for-the-badge&logo=spring&logoColor=white">

### AI/ML Framework

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/Transformers-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white">
<img src="https://img.shields.io/badge/Sentence_Transformers-4285F4?style=for-the-badge">

### AI Models & Systems

<img src="https://img.shields.io/badge/text--embedding--3-Embedding-FF6F00?style=for-the-badge">
<img src="https://img.shields.io/badge/gpt--5--nano-LLM-4285F4?style=for-the-badge">
<img src="https://img.shields.io/badge/RAG-System-00C853?style=for-the-badge">

### Data Processing

<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/CSV-Database-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white">

### DevOps & Deployment

<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
<img src="https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white">

### CI/CD

<img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white">
<img src="https://img.shields.io/badge/azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white">
<img src="https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white">

</div>

---

## 🏗️ 상세 구현 내용

### 📊 서비스 플로우

<div align="center">
  <img src="./ops/images/서비스 플로우.png" alt="서비스 플로우" width="800"/>
</div><br>

### 📋 시스템 아키텍처

<div align="center">
  <img src="./ops/images/시스템 아키텍처.png" alt="시스템 아키텍처" width="800"/>
</div><br>

### 📊 ERD

<div align="center">
  <img src="./ops/images/ERD.png" alt="ERD" width="800"/>
</div><br>

### 📊 AI 모델 플로우

<div align="center">
  <img src="./ops/images/AI 모델 플로우.png" alt="AI 모델 플로우.png" width="800"/>
</div><br>

---

# 🚀 배포 가이드

## 🚀 소스코드 / 이미지기반 가이드

- 📖 **[소스코드 기반- 설치 및 실행 가이드](https://github.com/LxNx-Hn/chatbot-with-kt-dgucenter/wiki/%EC%86%8C%EC%8A%A4%EC%BD%94%EB%93%9C%EA%B8%B0%EB%B0%98-%EB%B0%B0%ED%8F%AC%EA%B0%80%EC%9D%B4%EB%93%9C)** - 시스템 요구사항, Docker 설치, 트러블슈팅 포함

- 📖 **[이미지 기반- 설치 및 실행 가이드](https://github.com/LxNx-Hn/chatbot-with-kt-dgucenter/wiki/%EC%9D%B4%EB%AF%B8%EC%A7%80-%EA%B8%B0%EB%B0%98-%EB%B0%B0%ED%8F%AC-%EA%B0%80%EC%9D%B4%EB%93%9C)** - 컨테이너 실행, 환경변수 주입 , 트러블슈팅 포함

---

## 🚀CI/CD (자동 배포 시스템)

### 🔄 자동 배포 파이프라인

GitHub Actions 기반의 완전 자동화된 빌드 및 배포 시스템을 구축했습니다.

- **태그 기반 배포**: `v*` 태그 푸시 시 자동 빌드/배포 실행
- **멀티 플랫폼 지원**: 백엔드(RunPod), 프론트엔드(Netlify) 동시 배포
- **컨테이너 이미지 관리**: GitHub Container Registry(GHCR) 활용
- **환경별 배포**: Production/Development 환경 분리

### 📦 배포 환경

<div align="center">
  <table>
    <tr>
      <th>구분</th>
      <th>플랫폼</th>
      <th>용도</th>
      <th>특징</th>
    </tr>
    <tr>
      <td><b>Backend</b></td>
      <td>RunPod</td>
      <td>GPU 지원 LLM 서비스</td>
      <td>자동 스케일링, 비용 최적화</td>
    </tr>
    <tr>
      <td><b>Frontend</b></td>
      <td>Netlify</td>
      <td>React 앱 호스팅</td>
      <td>CDN, 자동 SSL, 무료 티어</td>
    </tr>
    <tr>
      <td><b>Container Images</b></td>
      <td>GitHub Container Registry</td>
      <td>Docker 이미지 저장소</td>
      <td>공개 이미지, 버전 관리</td>
    </tr>
  </table>
</div>

### 🛠️ CI/CD 워크플로우

**📋 [GitHub Actions 워크플로우 코드](https://github.com/LxNx-Hn/chatbot-with-kt-dgucenter/blob/main/.github/workflows/deploy.yml)** - 전체 빌드/배포 자동화 스크립트

**📖 [CI/CD 가이드문서](https://github.com/LxNx-Hn/chatbot-with-kt-dgucenter/wiki/CI-CD-%EA%B0%80%EC%9D%B4%EB%93%9C%EB%AC%B8%EC%84%9C)** - 워크플로우 설정, 사용방법 ,트러블슈팅 포함

<details>
<summary><b>🔧 워크플로우 주요 단계</b></summary>

1. **코드 체크아웃** - 최신 소스코드 가져오기
2. **Docker 이미지 빌드** - 백엔드/프론트엔드 컨테이너 빌드
3. **GHCR 푸시** - 빌드된 이미지 레지스트리에 업로드
4. **RunPod 배포** - 백엔드 API 자동 배포
5. **Netlify 배포** - 프론트엔드 자동 배포 (병렬 처리)

**🎯 특징:**

- 주석 기반 유연한 배포 설정 (다양한 플랫폼 지원 준비)
- 환경변수 보안 관리 (GitHub Secrets 활용)
- 실패 시 자동 롤백 지원

</details>

### 🔗 배포된 서비스 및 이미지

- **🌐 프론트엔드**: [Netlify 서비스](https://your-app.netlify.app)
- **⚙️ 백엔드 API**: [RunPod 서비스](https://your-api.runpod.io)
- **🐳 Docker 이미지**:
- Backend: [`ghcr.io/lxnx-hn/chatbot-with-kt-dgucenter-backend:latest`](https://github.com/LxNx-Hn/chatbot-with-kt-dgucenter/pkgs/container/chatbot-with-kt-dgucenter-backend)
- Frontend: [`ghcr.io/lxnx-hn/chatbot-with-kt-dgucenter-frontend:latest`](https://github.com/LxNx-Hn/chatbot-with-kt-dgucenter/pkgs/container/chatbot-with-kt-dgucenter-frontend)

## 👨‍💻 팀원별 상세 업무

<details>
<summary><b>🎯 전보윤 - ProductManager, AI</b></summary>

### 담당 업무

- **프로젝트 총괄 관리**
- **요구사항 분석 및 우선순위 결정**
- **API 관리 및 회계 처리**
- **시스템 검토 및 피드백 반영**
- **프론트엔드 진행 상황 점검 및 반응형 UI 디자인 설계**
- **외부 커뮤니케이션**
- **최종 발표 준비**
- **코드 의존성관리 및 구조화**

### 주요 성과

- **체계적 프로젝트 관리**: 요구사항 우선순위 및 예산 검토를 통해 불필요한 자원 소모 최소화
- **서비스 품질 개선**: 피드백 반영을 통한 시스템 개선 및 DB 지원으로 정확성 강화
- **효율적 예산 운영**: 예산 우선순위 관리 및 회계 처리의 투명성을 확보
- **내외부 커뮤니케이션 강화**: 팀 내부 협업과 유관기관과의 원활한 소통을 통해 프로젝트 완성도 제고
- **성공적 최종 발표**: 프로젝트의 진행 과정과 주요 성과를 체계적으로 정리하여 워크샵에서 효과적으로 공유
- **UX/UI 설계** : 모바일등의 기기에서도 원활한 작동이 가능하도록 React기반 반응형 UI 설계

### 사용 기술

**Backend & AI**

- `RAG (Retrieval-Augmented Generation)`

**Project Management & Collaboration**

- `Notion`, `Git`

**Design & Documentation**

- `Figma` , `Adove PremierPro`
- `Google Sheets / Excel`

**Frontend Review**

- `React`
- `JavaScript`
- `CSS`

**Version Control**

- `Git`, `GitHub`

</details>

<details>
<summary><b>🏗️ 강버들 - FrontEnd</b></summary>

### 담당 업무

- **프로젝트 전체 설계 및 아키텍처 구성**
- **백엔드 개발 지원 및 FastAPI 서버 구축**
- **프론트엔드 UI 구성 및 React 컴포넌트 개발**
- **배포 환경 구축 및 산출물 관리**
- **회의록 정리 및 진행상황 기록 관리**
- **팀원 기술 지원**
- **각종 프로젝트 문서 작성 및 관리**

### 주요 성과

- **RAG 시스템 도입**: 임베딩 모델 기반 검색 증강 생성 구조 설계 및 구현
- **전체 아키텍처 구축**: 프론트엔드-백엔드 연결 및 시스템 전반 설계
- **CI/CD 파이프라인 구축**: GitHub Actions를 통한 자동화된 배포 시스템 구현
- **컨테이너화 및 배포**: Docker 환경 구성과 Netlify, Runpod 배포 인프라 구축
- **프로젝트 관리**: 체계적인 일정 관리와 성과 기록을 통한 효율적 개발 진행

### 사용 기술

**Project Management**

- `Notion`, `Git` , `Github Project` , `GithubWiki`

**Backend & AI**

- `Python`, `FastAPI`, `PyTorch`
- `Transformers`, `Sentence-Transformers`
- `RAG (Retrieval-Augmented Generation)`

**Frontend**

- `React`, `JavaScript`, `CSS3`

**DevOps**

- `Docker`, `Docker Compose`, `YML`
- `GitHub Actions`, `Netlify`, `Runpod`

**Version Control**

- `Git`, `GitHub`

</details>

<details>
<summary><b>⚡ 김형모 - BackEnd</b></summary>

### 담당 업무

- **챗봇 인터페이스 개발**
- **데이터 전처리 보조 및 API 연동**
- **챗봇 최적화를 위한 프롬프트 엔지니어링**

### 주요 성과

- **핵심기능 개발** : 창업, 정책, 트렌드의 3가지 챗봇 핵심기능 개발
- **프롬포트 최적화**: 사용자가 정보를 쉽게 습득할 수 있도록 출력 프롬프트 최적화
- **하이퍼 파라미터 튜닝**: 적절한 출력을 위한 모델 하이퍼 파라미터 미세조정
- **할루시네이션 최소화**: 모델이 전처리한 데이터가 아닌 스스로 생성한 데이터로 이상한 정보 출력하는 것을 방지
- **FastAPI활용 백엔드 서비스 구축**: uvicorn, fastapi 활용 백엔드 서버 구현

### 사용 기술

**Backend & AI**

- `Python`, `FastAPI`, `PyTorch`
- `Transformers`, `Sentence-Transformers`
- `RAG (Retrieval-Augmented Generation)`

**Frontend**

- `React`

**DevOps**

- `Docker`

**Version Control**

- `Git`, `GitHub`

</details>

<details>

<summary><b>🗄️ 공동욱 - BackEnd</b></summary>

### 담당 업무

- **데이터 탐색(EDA) 및 확보 가능한 데이터 확정**
- **데이터 정제 로직 개발** (중복, 이상치, 결측치 처리)
- **RAG 모델 최적화를 위한 데이터셋 가공 및 프롬프트 테스트**
- **FastAPI 기반 데이터 분석 API 서버 개발**
- **프로젝트 데이터 관련 기술 문서 작성 및 관리**

### 주요 성과

- **데이터 파이프라인 구축**: 공공 인허가 데이터의 수집, 정제, 검증 과정을 자동화하여 데이터 신뢰도 확보 및 처리 시간 단축
- **RAG 모델 성능 향상**: 고품질 데이터셋 제공과 프롬프트 최적화를 통해 AI 모델의 검색 정확도 및 답변 품질 개선에 기여
- **데이터 분석 API 개발**: FastAPI 기반 트렌드 분석 서비스 초기 모델 설계 및 구현
- **파이프라인 모니터링 시스템 개발** : DB->LLM 데이터 전달 모니터링 알고리즘 구현 및 적용, 정확한 데이터 전달에 기여
- **데이터 유의어 처리 및 문장화** : CSV기반 데이터 컬럼 유의어 탐색 시스템 구현 및 데이터 문장화 시스템 구현

### 사용 기술

**Data, Backend & AI**

- `Python`, `Pandas`, `NumPy`, `FastAPI`,
- `RAG (Retrieval-Augmented Generation)`

**Tools & Version Control**

- `Git`, `GitHub`, `Google Colab`, `CSV`

**Version Control**

- `Git`, `GitHub`

</details>

<details>
<summary><b>🔗 윤성빈 - AI</b></summary>

### 담당 업무

- **시스템 연동 및 데이터 파이프라인 설계**
- **분류 모델 성능 개선 및 운영**
- **데이터 구축 및 정제**
- **프롬프트 하이퍼파라미터 튜닝**
- **임베딩모듈 구현**

### 주요 성과

- **분류 모델 하이퍼 파라미터 튜닝** : 하이퍼파라미터, 프롬포트 미세조정을 통해 정확도 14%향상, F-1 Score 8%향상
- **데이터 전처리를 통한 임베딩시스템 구현** : 데이터 가공 및 키워드 추출을 통한 임베딩 효율 향상 및 답변 일관성 유지
- **키워드 추출모델 제작** : 질문에서 키워드를 추출할수있는 모델 제작 및 프롬포트 엔지니어링
- **핵심기능별 임베딩 로직 구체화** : 3가지 핵심기능별 맞춤형 데이터 전처리 기능 개발
- **계층적 검색 시스템 구축** : 핵심 업종 파악 후 LLM응답에 벡터 검색 결과를 추가하는 계층적 검색 방식 구현

### 사용 기술

**AI & Data**

- `Python`, `Pandas`, `PyTorch`
- `Transformers`, `Sentence-Transformers`

**Data Processing**

- `Regex`, `CSV`

**Version Control**

- `Git`, `GitHub`

</details>

<details>
<summary><b>🏛️ 이세용 - BackEnd</b></summary>

### 담당 업무

- **시스템 연동 및 데이터 파이프라인 설계**
- **분류 모델 성능 개선 및 운영**
- **데이터 구축 및 정제**
- **프롬프트 하이퍼파라미터 튜닝**
- **임베딩모듈 구현**

### 주요 성과

- **분류 모델 하이퍼 파라미터 튜닝** : 하이퍼파라미터, 프롬포트 미세조정을 통해 정확도 14%향상, F-1 Score 8%향상
- **데이터 전처리를 통한 임베딩시스템 구현** : 데이터 가공 및 키워드 추출을 통한 임베딩 효율 향상 및 답변 일관성 유지
- **키워드 추출모델 제작** : 질문에서 키워드를 추출할수있는 모델 제작 및 프롬포트 엔지니어링
- **핵심기능별 임베딩 로직 구체화** : 3가지 핵심기능별 맞춤형 데이터 전처리 기능 개발
- **계층적 검색 시스템 구축** : 핵심 업종 파악 후 LLM응답에 벡터 검색 결과를 추가하는 계층적 검색 방식 구현

### 사용 기술

**AI & Data**

- `Python`, `Pandas`, `PyTorch`
- `Transformers`, `Sentence-Transformers`

**Data Processing**

- `Regex`, `CSV`

**Version Control**

- `Git`, `GitHub`

</details>

<details>
<summary><b>🚧 장우진 - AI</b></summary>

### 담당 업무

- **시스템 연동 및 데이터 파이프라인 설계**
- **분류 모델 성능 개선 및 운영**
- **데이터 구축 및 정제**
- **프롬프트 하이퍼파라미터 튜닝**
- **임베딩모듈 구현**

### 주요 성과

- **분류 모델 하이퍼 파라미터 튜닝** : 하이퍼파라미터, 프롬포트 미세조정을 통해 정확도 14%향상, F-1 Score 8%향상
- **데이터 전처리를 통한 임베딩시스템 구현** : 데이터 가공 및 키워드 추출을 통한 임베딩 효율 향상 및 답변 일관성 유지
- **키워드 추출모델 제작** : 질문에서 키워드를 추출할수있는 모델 제작 및 프롬포트 엔지니어링
- **핵심기능별 임베딩 로직 구체화** : 3가지 핵심기능별 맞춤형 데이터 전처리 기능 개발
- **계층적 검색 시스템 구축** : 핵심 업종 파악 후 LLM응답에 벡터 검색 결과를 추가하는 계층적 검색 방식 구현

### 사용 기술

**AI & Data**

- `Python`, `Pandas`, `PyTorch`
- `Transformers`, `Sentence-Transformers`

**Data Processing**

- `Regex`, `CSV`

**Version Control**

- `Git`, `GitHub`

</details>

<details>
<summary><b>🏢 황도윤 - FrontEnd</b></summary>

### 담당 업무

- **시스템 연동 및 데이터 파이프라인 설계**
- **분류 모델 성능 개선 및 운영**
- **데이터 구축 및 정제**
- **프롬프트 하이퍼파라미터 튜닝**
- **임베딩모듈 구현**

### 주요 성과

- **분류 모델 하이퍼 파라미터 튜닝** : 하이퍼파라미터, 프롬포트 미세조정을 통해 정확도 14%향상, F-1 Score 8%향상
- **데이터 전처리를 통한 임베딩시스템 구현** : 데이터 가공 및 키워드 추출을 통한 임베딩 효율 향상 및 답변 일관성 유지
- **키워드 추출모델 제작** : 질문에서 키워드를 추출할수있는 모델 제작 및 프롬포트 엔지니어링
- **핵심기능별 임베딩 로직 구체화** : 3가지 핵심기능별 맞춤형 데이터 전처리 기능 개발
- **계층적 검색 시스템 구축** : 핵심 업종 파악 후 LLM응답에 벡터 검색 결과를 추가하는 계층적 검색 방식 구현

### 사용 기술

**AI & Data**

- `Python`, `Pandas`, `PyTorch`
- `Transformers`, `Sentence-Transformers`

**Data Processing**

- `Regex`, `CSV`

**Version Control**

- `Git`, `GitHub`

</details>

---

## 📝 회의록 및 개발 과정

**[📊 GitHub 프로젝트 보드](https://github.com/users/LxNx-Hn/projects/3)** - 실시간 개발 현황, 회의록, 이슈 트래킹

---

## 💬 프로젝트 소감

### 🎯 전보윤 - ProductManager, AI

> "사용자 입장에서 결과값을 받았을 때 어떻게 하면 신뢰할 수 있는지 고민해볼 수 있었던 프로젝트였습니다. 
> 그 결과 조건부 분위수 함수로부터 확률 추출이라는 문제를 직면했고,  부족했던 부분을 공부해볼 수 있어서 좋았습니다. 
> 다만, 데이터를 깊게 이해하고 더 좋은 품질의 데이터로 가공을 진행하기 어려웠던 점이 아쉬웠습니다."

### 🏗️ 강버들 - FrontEnd

> "프로젝트를 진행하면서 느낀 점이나 소감을 작성해주세요."

### ⚡ 김형모 - BackEnd

> "프로젝트를 진행하면서 느낀 점이나 소감을 작성해주세요."

### 🗄️ 공동욱 - BackEnd

> "프로젝트를 진행하면서 느낀 점이나 소감을 작성해주세요."

### 🔗 윤성빈 - AI

> "프로젝트를 진행하면서 느낀 점이나 소감을 작성해주세요."

### 🏛️ 이세용 - BackEnd

> "프로젝트를 진행하면서 느낀 점이나 소감을 작성해주세요."

### 🚧 장우진 - AI

> "걱정이 앞선 시작이었지만, 부족한 저를 이끌어준 팀원들 덕분에 혼란 속에서도 많은 것을 배우고 채울 수 있었습니다. 서툰 저와 함께하며 성장의 기회를 준 여러분께 깊은 감사를 전합니다."

### 🏢 황도윤 - FrontEnd

> "프로젝트를 진행하면서 느낀 점이나 소감을 작성해주세요."

---

<div align="center">
  <sub>Built with ❤️ AIVLE 스쿨 빅 프로젝트 16조</sub>
</div>
