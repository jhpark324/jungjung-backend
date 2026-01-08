"""
LangGraph 기반 RAG 파이프라인 (시각화 병렬 처리 포함)
======================================================
HyDE → Retriever → (Generator || Visualizer) → Aggregate

그래프 흐름:
    START
      │
      ▼
    [HyDE 노드] → 가상 문서 생성
      │
      ▼
    [Retriever 노드] → 앙상블 검색 + Parent 확장
      │
      ├──────────────────┐
      ▼                  ▼
    [Generator]     [Visualizer]
      │                  │
      └────────┬─────────┘
               ▼
         [Aggregate 노드]
               │
               ▼
              END
"""

from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from backend.rag.hyde import HyDEGenerator
from backend.rag.retriever import EnsembleParentChildRetriever
from backend.rag.generator import AnswerGenerator
from backend.rag.visualization import EvidenceVisualizer


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
    enable_visualization: bool  # 시각화 활성화 여부
    
    # 중간 결과
    hypothetical_doc: Optional[str]
    search_query: str
    documents: List[Document]
    
    # 출력 - 답변
    answer: str
    
    # 출력 - 시각화
    visualization: Optional[Dict[str, Any]]


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


def create_visualizer_node(visualizer: EvidenceVisualizer):
    """Visualizer 노드 생성"""
    
    def visualizer_node(state: RAGState) -> dict:
        """시각화 생성 노드"""
        if not state.get("enable_visualization", False):
            return {"visualization": None}
        
        documents = state.get("documents", [])
        if not documents:
            return {"visualization": None}
        
        # 쿼리 기반 고유 ID 생성
        import hashlib
        from datetime import datetime
        query_hash = hashlib.md5(state["query"].encode()).hexdigest()[:8]
        query_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query_hash}"
        
        result = visualizer.visualize(documents, query_id)
        
        return {"visualization": result}
    
    return visualizer_node


def create_aggregate_node():
    """결과 집계 노드 생성"""
    
    def aggregate_node(state: RAGState) -> dict:
        """결과 집계 - 모든 병렬 노드 완료 후 실행"""
        # 이미 state에 answer와 visualization이 있음
        # 추가 처리가 필요하면 여기서 수행
        return {}
    
    return aggregate_node


# =============================================================================
# 그래프 빌드
# =============================================================================
def build_rag_graph(
    retriever: EnsembleParentChildRetriever,
    hyde_generator: HyDEGenerator,
    answer_generator: AnswerGenerator,
    visualizer: Optional[EvidenceVisualizer] = None
) -> StateGraph:
    """
    RAG 그래프 빌드
    
    흐름: START → HyDE → Retriever → (Generator || Visualizer) → Aggregate → END
    """
    # 그래프 생성
    graph = StateGraph(RAGState)
    
    # 노드 추가
    graph.add_node("hyde", create_hyde_node(hyde_generator))
    graph.add_node("retrieve", create_retriever_node(retriever))
    graph.add_node("generate", create_generator_node(answer_generator))
    
    if visualizer:
        graph.add_node("visualize", create_visualizer_node(visualizer))
        graph.add_node("aggregate", create_aggregate_node())
    
    # 엣지 연결
    graph.add_edge(START, "hyde")
    graph.add_edge("hyde", "retrieve")
    
    if visualizer:
        # Retriever 이후 병렬 분기
        graph.add_edge("retrieve", "generate")
        graph.add_edge("retrieve", "visualize")
        # 병렬 노드들이 aggregate로 합류
        graph.add_edge("generate", "aggregate")
        graph.add_edge("visualize", "aggregate")
        graph.add_edge("aggregate", END)
    else:
        # 시각화 없이 기존 흐름
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
        answer_generator: AnswerGenerator,
        visualizer: Optional[EvidenceVisualizer] = None
    ):
        self.visualizer = visualizer
        self.graph = build_rag_graph(
            retriever, 
            hyde_generator, 
            answer_generator,
            visualizer
        )
    
    def invoke(
        self,
        query: str,
        use_hyde: bool = True,
        top_children: int = 20,
        top_parents: int = 5,
        enable_visualization: bool = True
    ) -> dict:
        """
        RAG 파이프라인 실행
        
        Args:
            query: 사용자 질문
            use_hyde: HyDE 사용 여부
            top_children: 검색할 child 문서 수
            top_parents: 반환할 parent 문서 수
            enable_visualization: 시각화 생성 여부
        
        Returns:
            {
                "query": str,
                "hypothetical_doc": str | None,
                "documents": List[Document],
                "answer": str,
                "visualization": dict | None
            }
        """
        initial_state = {
            "query": query,
            "use_hyde": use_hyde,
            "top_children": top_children,
            "top_parents": top_parents,
            "enable_visualization": enable_visualization and self.visualizer is not None,
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "query": result["query"],
            "hypothetical_doc": result.get("hypothetical_doc"),
            "documents": result["documents"],
            "answer": result["answer"],
            "visualization": result.get("visualization")
        }
