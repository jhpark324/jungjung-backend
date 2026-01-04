"""
Ensemble Parent-Child Retriever with BM25 + VectorStore
========================================================
LangChain의 EnsembleRetriever를 활용하여 BM25(키워드)와 VectorStore(의미)를 
앙상블로 결합하고, Child 검색 결과를 Parent로 확장하는 리트리버입니다.

구조:
    쿼리
    ├── VectorStore(Dense) → Child들 (doc_id 포함)
    └── BM25(Sparse) → Child들 (doc_id 포함)
                │
                ▼
        EnsembleRetriever (RRF) → 최종 Child 결과
                │
                ▼
        doc_id → DocStore → Parent 반환
"""

import os
import pickle
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever


# =============================================================================
# 1. 저장소 경로 설정
# =============================================================================
CHILD_STORE_PATH = "/Users/parkjehyeong/jungjung/backend/vector_store/child_store"
PARENT_STORE_PATH = "/Users/parkjehyeong/jungjung/backend/vector_store/parent_store"
BM25_INDEX_PATH = "/Users/parkjehyeong/jungjung/backend/vector_store/bm25_index.pkl"


# =============================================================================
# 2. 앙상블 Parent-Child Retriever 클래스 (LangChain EnsembleRetriever 활용)
# =============================================================================
class EnsembleParentChildRetriever:
    """
    LangChain EnsembleRetriever + Parent 확장을 수행하는 래퍼 클래스
    
    - 앙상블(RRF): LangChain의 EnsembleRetriever 사용
    - Parent 확장: doc_id로 DocStore에서 Parent 조회
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
        """
        Child의 doc_id로 Parent 문서 조회 (중복 제거)
        """
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
        """
        앙상블 검색 + Parent 확장
        
        Args:
            query: 검색 쿼리
            top_children: 앙상블에서 가져올 Child 수
            top_parents: 최종 반환할 Parent 수
        
        Returns:
            Parent 문서 리스트
        """
        # 1. EnsembleRetriever로 앙상블 검색 (BM25 + Dense, RRF 적용)
        self.vector_retriever.search_kwargs = {"k": top_children}
        self.bm25_retriever.k = top_children
        
        fused_children = self.ensemble_retriever.invoke(query)
        
        # 2. Parent 확장
        parents = self._expand_to_parents(fused_children)
        
        return parents[:top_parents]


# =============================================================================
# 3. BM25 인덱스 생성/로드 함수
# =============================================================================
def get_child_documents_from_vectorstore(vectorstore: Chroma) -> List[Document]:
    """
    VectorStore에서 모든 Child 문서 추출
    """
    # Chroma에서 모든 데이터 가져오기
    collection = vectorstore._collection
    result = collection.get(include=["documents", "metadatas"])
    
    documents = []
    for i, content in enumerate(result["documents"]):
        metadata = result["metadatas"][i] if result["metadatas"] else {}
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    print(f"VectorStore에서 {len(documents)}개 Child 문서 추출됨")
    return documents


def create_or_load_bm25_retriever(vectorstore: Chroma, force_recreate: bool = False) -> BM25Retriever:
    """
    BM25 인덱스 생성 또는 로드
    """
    if os.path.exists(BM25_INDEX_PATH) and not force_recreate:
        print("기존 BM25 인덱스 로드 중...")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
        print("BM25 인덱스 로드 완료")
    else:
        print("BM25 인덱스 생성 중...")
        child_docs = get_child_documents_from_vectorstore(vectorstore)
        bm25_retriever = BM25Retriever.from_documents(child_docs)
        
        # 저장
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print(f"BM25 인덱스 저장 완료: {BM25_INDEX_PATH}")
    
    return bm25_retriever


# =============================================================================
# 4. 앙상블 리트리버 초기화 함수
# =============================================================================
def setup_ensemble_retriever(force_recreate_bm25: bool = False) -> EnsembleParentChildRetriever:
    """
    앙상블 리트리버 초기화
    """
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
    
    # 4. BM25 인덱스 생성/로드
    bm25_retriever = create_or_load_bm25_retriever(vectorstore, force_recreate_bm25)
    
    # 5. 앙상블 리트리버 생성
    ensemble_retriever = EnsembleParentChildRetriever(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        docstore=docstore,
        dense_weight=0.5,   # 의미 검색 70%
        sparse_weight=0.5   # 키워드 검색 30%
    )
    
    print("앙상블 리트리버 초기화 완료!")
    return ensemble_retriever


# =============================================================================
# 5. 검색 테스트
# =============================================================================
def test_retrieval(retriever: EnsembleParentChildRetriever, query: str, top_parents: int = 3):
    """
    검색 테스트
    """
    print(f"\n{'='*60}")
    print(f"쿼리: {query}")
    print(f"{'='*60}")
    
    results = retriever.invoke(query, top_children=20, top_parents=top_parents)
    
    print(f"\n검색 결과: {len(results)}개의 Parent 문서 반환\n")
    
    for i, doc in enumerate(results, 1):
        print(f"[결과 {i}]")
        print(f"  메타데이터: {doc.metadata}")
        print(f"  내용 (앞 300자):")
        print(f"  {doc.page_content[:300]}...")
        print()
    
    return results


# =============================================================================
# 6. 메인 실행
# =============================================================================
if __name__ == "__main__":
    # 앙상블 리트리버 초기화
    retriever = setup_ensemble_retriever(force_recreate_bm25=True)
    
    # 검색 테스트
    test_queries = [
        "한지아 의원이 대표발의한 법안은?",
        "12월 2일 회의",
        "검역법 일부개정법률안",
        "수도권 규제"
    ]
    
    for query in test_queries:
        test_retrieval(retriever, query)
