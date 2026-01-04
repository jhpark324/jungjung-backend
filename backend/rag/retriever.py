"""
Ensemble Parent-Child Retriever with BM25 + VectorStore
========================================================
LangChain의 EnsembleRetriever를 활용하여 BM25(키워드)와 VectorStore(의미)를 
앙상블로 결합하고, Child 검색 결과를 Parent로 확장하는 리트리버입니다.
"""

import os
import pickle
from typing import List
from pathlib import Path

from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever


# =============================================================================
# 저장소 경로 설정
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
CHILD_STORE_PATH = str(BASE_DIR / "vector_store" / "child_store")
PARENT_STORE_PATH = str(BASE_DIR / "vector_store" / "parent_store")
BM25_INDEX_PATH = str(BASE_DIR / "vector_store" / "bm25_index.pkl")


# =============================================================================
# 앙상블 Parent-Child Retriever 클래스
# =============================================================================
class EnsembleParentChildRetriever:
    """
    LangChain EnsembleRetriever + Parent 확장을 수행하는 래퍼 클래스
    """
    
    def __init__(
        self, 
        vectorstore: Chroma, 
        bm25_retriever: BM25Retriever,
        docstore,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.docstore = docstore
        
        # VectorStore를 retriever로 변환
        self.vector_retriever = vectorstore.as_retriever()
        
        # LangChain EnsembleRetriever 생성 (RRF 내장)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[dense_weight, sparse_weight]
        )
    
    def _expand_to_parents(self, children: List[Document]) -> List[Document]:
        """Child의 doc_id로 Parent 문서 조회 (중복 제거)"""
        seen_ids = set()
        parents = []
        
        for child in children:
            doc_id = child.metadata.get("doc_id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                try:
                    parent = self.docstore.mget([doc_id])[0]
                    if parent:
                        parents.append(parent)
                except Exception as e:
                    print(f"Parent 조회 실패 (doc_id: {doc_id}): {e}")
        
        return parents
    
    def invoke(self, query: str, top_children: int = 20, top_parents: int = 5) -> List[Document]:
        """앙상블 검색 + Parent 확장"""
        self.vector_retriever.search_kwargs = {"k": top_children}
        self.bm25_retriever.k = top_children
        
        fused_children = self.ensemble_retriever.invoke(query)
        parents = self._expand_to_parents(fused_children)
        
        return parents[:top_parents]


# =============================================================================
# 리트리버 초기화 함수
# =============================================================================
def create_ensemble_retriever(
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5
) -> EnsembleParentChildRetriever:
    """앙상블 리트리버 초기화"""
    print("앙상블 리트리버 초기화 중...")
    
    # 1. Embedding 모델
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large-query")
    
    # 2. VectorStore 로드
    vectorstore = Chroma(
        collection_name="parliament_child_chunks",
        embedding_function=embeddings,
        persist_directory=CHILD_STORE_PATH
    )
    print(f"VectorStore 로드 완료: {vectorstore._collection.count()}개 Child")
    
    # 3. DocStore 로드 (Parent용)
    base_store = LocalFileStore(PARENT_STORE_PATH)
    docstore = create_kv_docstore(base_store)
    print("DocStore 로드 완료")
    
    # 4. BM25 인덱스 로드
    if os.path.exists(BM25_INDEX_PATH):
        print("기존 BM25 인덱스 로드 중...")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
        print("BM25 인덱스 로드 완료")
    else:
        raise FileNotFoundError(f"BM25 인덱스를 찾을 수 없습니다: {BM25_INDEX_PATH}")
    
    # 5. 앙상블 리트리버 생성
    retriever = EnsembleParentChildRetriever(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        docstore=docstore,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight
    )
    
    print("앙상블 리트리버 초기화 완료!")
    return retriever
