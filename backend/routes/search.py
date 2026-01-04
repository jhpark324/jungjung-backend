"""
검색 API 라우트
===============
"""

from fastapi import APIRouter, Request, Query
from typing import Dict, Any

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search")
async def search_documents(
    request: Request,
    q: str = Query(..., description="검색 쿼리"),
    limit: int = Query(default=5, ge=1, le=20, description="반환할 결과 수"),
    use_hyde: bool = Query(default=True, description="HyDE 사용 여부 (RAG 모드에서만)")
) -> Dict[str, Any]:
    """
    에이전트 기반 검색 API
    
    라우터가 질문을 분류:
    - **국회 회의록 관련** → RAG 에이전트 (HyDE + 검색 + 답변 생성)
    - **일반 질문** → 일반 LLM 응답
    
    Parameters:
    - **q**: 질문/검색 쿼리
    - **limit**: 반환할 문서 수 (RAG 모드에서만 적용)
    - **use_hyde**: HyDE 사용 여부 (RAG 모드에서만 적용)
    """
    rag_service = request.app.state.rag_service
    result = rag_service.search(query=q, top_parents=limit, use_hyde=use_hyde)
    
    response = {
        "query": q,
        "route": result["route"],
        "answer": result["answer"],
    }
    
    # RAG 모드인 경우 추가 정보
    if result["route"] == "RAG":
        response["hypothetical_doc"] = result["hypothetical_doc"]
        response["source_count"] = len(result["sources"]) if result["sources"] else 0
        response["sources"] = result["sources"]
    
    return response
