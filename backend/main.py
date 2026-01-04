from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.rag.retriever import create_ensemble_retriever
from backend.rag.hyde import HyDEGenerator
from backend.rag.generator import AnswerGenerator
from backend.rag.graph import RAGGraph
from backend.rag.router import QuestionRouter
from backend.rag.agent import Agent, GeneralLLM
from backend.services.rag_service import RAGService
from backend.routes.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 시작/종료 시 실행되는 lifespan 이벤트
    """
    print("🚀 서버 시작: 에이전트 초기화 중...")
    
    # 1. 리트리버 초기화
    print("  📦 앙상블 리트리버 로드 중...")
    retriever = create_ensemble_retriever(
        dense_weight=0.5,
        sparse_weight=0.5
    )
    
    # 2. HyDE 생성기 초기화
    print("  🔮 HyDE 생성기 초기화 중...")
    hyde_generator = HyDEGenerator(model_name="solar-pro2")
    
    # 3. 답변 생성기 초기화
    print("  💬 답변 생성기 초기화 중...")
    answer_generator = AnswerGenerator(model_name="solar-pro2")
    
    # 4. RAG 그래프 빌드
    print("  🔗 RAG 그래프 빌드 중...")
    rag_graph = RAGGraph(retriever, hyde_generator, answer_generator)
    
    # 5. 라우터 및 일반 LLM 초기화
    print("  🔀 라우터 초기화 중...")
    question_router = QuestionRouter(model_name="solar-pro2")
    general_llm = GeneralLLM(model_name="solar-pro2")
    
    # 6. 에이전트 빌드
    print("  🤖 에이전트 빌드 중...")
    agent = Agent(question_router, rag_graph, general_llm)
    
    # 7. 서비스 등록
    app.state.rag_service = RAGService(agent)
    print("✅ 에이전트 준비 완료! (Router + RAG + LLM)")
    
    yield
    
    print("👋 서버 종료")


app = FastAPI(
    title="JungJung Agent API",
    description="LangGraph 기반 에이전트 API (라우터 + RAG + LLM)",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(search_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}