"""
RAG 서비스 레이어
=================
에이전트 기반 RAG 서비스 - 라우터가 질문을 분류하여 적절한 처리
"""

from typing import Dict, Any
from backend.rag.agent import Agent


class RAGService:
    """에이전트 기반 RAG 서비스"""
    
    def __init__(self, agent: Agent):
        self.agent = agent
    
    def search(
        self, 
        query: str, 
        top_children: int = 20,
        top_parents: int = 5,
        use_hyde: bool = True
    ) -> Dict[str, Any]:
        """
        에이전트 실행 - 질문 분류 후 적절한 처리
        
        Args:
            query: 검색 쿼리
            top_children: 앙상블에서 가져올 Child 수
            top_parents: 최종 반환할 Parent 수
            use_hyde: HyDE 사용 여부 (RAG 모드에서만 적용)
        
        Returns:
            에이전트 실행 결과
        """
        result = self.agent.invoke(
            query=query,
            use_hyde=use_hyde,
            top_children=top_children,
            top_parents=top_parents
        )
        
        # 응답 구성
        response = {
            "original_query": query,
            "route": result["route"],
            "answer": result["answer"],
        }
        
        # RAG 모드인 경우 추가 정보 포함
        if result["route"] == "RAG":
            response["hypothetical_doc"] = result["hypothetical_doc"]
            response["sources"] = [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in (result["documents"] or [])
            ]
        else:
            response["hypothetical_doc"] = None
            response["sources"] = None
        
        return response
