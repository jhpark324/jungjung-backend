"""
LangGraph 기반 RAG 파이프라인
============================
HyDE → Retriever → Generator 노드를 연결한 그래프 구조

그래프 흐름:
    START
      │
      ▼
    [HyDE 노드] → 가상 문서 생성
      │
      ▼
    [Retriever 노드] → 앙상블 검색 + Parent 확장
      │
      ▼
    [Generator 노드] → 답변 생성
      │
      ▼
    END (결과 반환)
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from backend.rag.hyde import HyDEGenerator
from backend.rag.retriever import EnsembleParentChildRetriever
from backend.rag.generator import AnswerGenerator


# =============================================================================
# State 스키마 정의
# =============================================================================
class RAGState(TypedDict):
    """RAG 파이프라인 상태"""
    # 입력
    query: str
    use_hyde: bool
    top_children: int
    top_parents: int
    
    # 중간 결과
    hypothetical_doc: Optional[str]
    search_query: str
    documents: List[Document]
    
    # 출력
    answer: str


# =============================================================================
# 노드 정의
# =============================================================================
def create_hyde_node(hyde_generator: HyDEGenerator):
    """HyDE 노드 생성"""
    
    def hyde_node(state: RAGState) -> dict:
        """가상 문서 생성 노드"""
        if state.get("use_hyde", True):
            hypothetical_doc = hyde_generator.generate(state["query"])
            search_query = f"{state['query']}\n\n{hypothetical_doc}"
        else:
            hypothetical_doc = None
            search_query = state["query"]
        
        return {
            "hypothetical_doc": hypothetical_doc,
            "search_query": search_query
        }
    
    return hyde_node


def create_retriever_node(retriever: EnsembleParentChildRetriever):
    """Retriever 노드 생성"""
    
    def retriever_node(state: RAGState) -> dict:
        """앙상블 검색 + Parent 확장 노드"""
        documents = retriever.invoke(
            query=state["search_query"],
            top_children=state.get("top_children", 20),
            top_parents=state.get("top_parents", 5)
        )
        
        return {"documents": documents}
    
    return retriever_node


def create_generator_node(generator: AnswerGenerator):
    """Generator 노드 생성"""
    
    def generator_node(state: RAGState) -> dict:
        """답변 생성 노드"""
        answer = generator.generate(
            question=state["query"],
            documents=state["documents"]
        )
        
        return {"answer": answer}
    
    return generator_node


# =============================================================================
# 그래프 빌드
# =============================================================================
def build_rag_graph(
    retriever: EnsembleParentChildRetriever,
    hyde_generator: HyDEGenerator,
    answer_generator: AnswerGenerator
) -> StateGraph:
    """
    RAG 그래프 빌드
    
    흐름: START → HyDE → Retriever → Generator → END
    """
    # 그래프 생성
    graph = StateGraph(RAGState)
    
    # 노드 추가
    graph.add_node("hyde", create_hyde_node(hyde_generator))
    graph.add_node("retrieve", create_retriever_node(retriever))
    graph.add_node("generate", create_generator_node(answer_generator))
    
    # 엣지 연결
    graph.add_edge(START, "hyde")
    graph.add_edge("hyde", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    # 컴파일
    return graph.compile()


# =============================================================================
# RAG 그래프 래퍼 클래스
# =============================================================================
class RAGGraph:
    """LangGraph 기반 RAG 파이프라인"""
    
    def __init__(
        self,
        retriever: EnsembleParentChildRetriever,
        hyde_generator: HyDEGenerator,
        answer_generator: AnswerGenerator
    ):
        self.graph = build_rag_graph(retriever, hyde_generator, answer_generator)
    
    def invoke(
        self,
        query: str,
        use_hyde: bool = True,
        top_children: int = 20,
        top_parents: int = 5
    ) -> dict:
        """
        RAG 파이프라인 실행
        
        Returns:
            {
                "query": str,
                "hypothetical_doc": str | None,
                "documents": List[Document],
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
            "hypothetical_doc": result.get("hypothetical_doc"),
            "documents": result["documents"],
            "answer": result["answer"]
        }
