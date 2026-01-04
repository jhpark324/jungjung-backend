"""
에이전트 그래프 (Agent Graph)
============================
라우터를 통해 질문을 분류하고 적절한 에이전트로 라우팅하는 상위 그래프

그래프 흐름:
    START
      │
      ▼
    [Router 노드] → 질문 분류 (RAG or LLM)
      │
      ├── RAG ──→ [RAG 에이전트] → 회의록 검색 + 답변 생성
      │                │
      └── LLM ──→ [LLM 노드] → 일반 LLM 응답
                       │
                       ▼
                      END
"""

from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.rag.router import QuestionRouter
from backend.rag.graph import RAGGraph


# =============================================================================
# State 스키마 정의
# =============================================================================
class AgentState(TypedDict):
    """에이전트 상태"""
    # 입력
    query: str
    use_hyde: bool
    top_children: int
    top_parents: int
    
    # 라우팅 결과
    route: Literal["RAG", "LLM"]
    
    # 출력 (RAG 또는 LLM)
    hypothetical_doc: Optional[str]
    documents: Optional[List[Document]]
    answer: str


# =============================================================================
# 일반 LLM 응답 클래스
# =============================================================================
GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 도움이 되는 AI 비서입니다.
사용자의 질문에 명확하고 정확하게 답변하세요.

참고: 국회 회의록 관련 질문은 별도의 전문 시스템에서 처리됩니다.
일반적인 질문에 대해 최선을 다해 답변해주세요."""),
    ("human", "{query}")
])


class GeneralLLM:
    """일반 LLM 응답 생성기"""
    
    def __init__(self, model_name: str = "solar-pro2"):
        self.llm = ChatUpstage(model=model_name)
        self.chain = GENERAL_PROMPT | self.llm | StrOutputParser()
    
    def generate(self, query: str) -> str:
        return self.chain.invoke({"query": query})


# =============================================================================
# 노드 정의
# =============================================================================
def create_router_node(router: QuestionRouter):
    """라우터 노드 생성"""
    
    def router_node(state: AgentState) -> dict:
        """질문 분류 노드"""
        route = router.route(state["query"])
        return {"route": route}
    
    return router_node


def create_rag_agent_node(rag_graph: RAGGraph):
    """RAG 에이전트 노드 생성"""
    
    def rag_agent_node(state: AgentState) -> dict:
        """RAG 파이프라인 실행 노드"""
        result = rag_graph.invoke(
            query=state["query"],
            use_hyde=state.get("use_hyde", True),
            top_children=state.get("top_children", 20),
            top_parents=state.get("top_parents", 5)
        )
        
        return {
            "hypothetical_doc": result["hypothetical_doc"],
            "documents": result["documents"],
            "answer": result["answer"]
        }
    
    return rag_agent_node


def create_llm_node(general_llm: GeneralLLM):
    """일반 LLM 노드 생성"""
    
    def llm_node(state: AgentState) -> dict:
        """일반 LLM 응답 노드"""
        answer = general_llm.generate(state["query"])
        
        return {
            "hypothetical_doc": None,
            "documents": None,
            "answer": answer
        }
    
    return llm_node


# =============================================================================
# 조건부 라우팅 함수
# =============================================================================
def route_question(state: AgentState) -> str:
    """라우팅 결정 함수"""
    return state["route"]


# =============================================================================
# 그래프 빌드
# =============================================================================
def build_agent_graph(
    router: QuestionRouter,
    rag_graph: RAGGraph,
    general_llm: GeneralLLM
) -> StateGraph:
    """
    에이전트 그래프 빌드
    
    흐름: START → Router → (RAG or LLM) → END
    """
    graph = StateGraph(AgentState)
    
    # 노드 추가
    graph.add_node("router", create_router_node(router))
    graph.add_node("rag_agent", create_rag_agent_node(rag_graph))
    graph.add_node("llm", create_llm_node(general_llm))
    
    # 엣지 연결
    graph.add_edge(START, "router")
    
    # 조건부 라우팅
    graph.add_conditional_edges(
        "router",
        route_question,
        {
            "RAG": "rag_agent",
            "LLM": "llm"
        }
    )
    
    graph.add_edge("rag_agent", END)
    graph.add_edge("llm", END)
    
    return graph.compile()


# =============================================================================
# 에이전트 래퍼 클래스
# =============================================================================
class Agent:
    """LangGraph 기반 에이전트"""
    
    def __init__(
        self,
        router: QuestionRouter,
        rag_graph: RAGGraph,
        general_llm: GeneralLLM
    ):
        self.graph = build_agent_graph(router, rag_graph, general_llm)
    
    def invoke(
        self,
        query: str,
        use_hyde: bool = True,
        top_children: int = 20,
        top_parents: int = 5
    ) -> dict:
        """
        에이전트 실행
        
        Returns:
            {
                "query": str,
                "route": "RAG" | "LLM",
                "hypothetical_doc": str | None,
                "documents": List[Document] | None,
                "answer": str
            }
        """
        initial_state = {
            "query": query,
            "use_hyde": use_hyde,
            "top_children": top_children,
            "top_parents": top_parents,
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "query": result["query"],
            "route": result["route"],
            "hypothetical_doc": result.get("hypothetical_doc"),
            "documents": result.get("documents"),
            "answer": result["answer"]
        }
