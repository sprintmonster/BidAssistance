from fastapi import FastAPI
# 다른 서버에서 오는 요청을 허용하기 위한 CORS 미들웨어
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 애플리케이션 생성 (API 문서 제목 설정)
app = FastAPI(title="ML API")

# 모든 출처(origin)에서의 접근을 허용하기 위한 설정
origins = ["*"]

# CORS(Cross-Origin Resource Sharing) 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 모든 도메인에서의 요청 허용
    allow_credentials=True,     # 쿠키, 인증 정보 포함 요청 허용
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # 허용할 HTTP 메서드
    allow_headers=origins,        # 모든 HTTP 헤더 허용
)

@app.get("/")
def read_root():
    return {"안녕": "FastAPI"}