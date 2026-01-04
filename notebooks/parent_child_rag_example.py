"""
Parent-Child Chunking Strategy Example with LangChain
=====================================================
LangChain의 ParentDocumentRetriever를 사용하여 Parent-Child 전략을 구현합니다.

핵심 개념:
- Parent: 큰 문맥 (예: 페이지 전체) -> DocStore에 저장
- Child: 작은 청크 (예: 500자) -> VectorStore에 저장 (검색용)
- 검색 시: Child를 검색하고, 해당 Child의 Parent를 반환
"""

import os
import re
import glob
import uuid
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 1. 필요한 모듈 임포트
# =============================================================================
from langchain_upstage import UpstageDocumentParseLoader, UpstageEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document


# =============================================================================
# 2. PDF 문서 로드 (Upstage Document Parser 사용)
# =============================================================================
def filter_complex_metadata(metadata: dict) -> dict:
    """
    ChromaDB가 지원하지 않는 복잡한 메타데이터(리스트, 딕셔너리)를 제거합니다.
    """
    filtered = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            filtered[key] = value
        # 리스트, 딕셔너리 등은 건너뜀
    return filtered


def load_all_pdfs_as_pages(pdf_dir: str):
    """
    디렉토리 내 모든 PDF를 페이지 단위로 로드합니다.
    각 Document = 1 페이지 (이것이 Parent가 됨)
    """
    pdf_files = glob.glob(f"{pdf_dir}/*.pdf")
    all_docs = []
    
    for pdf_path in pdf_files:
        print(f"로딩 중: {pdf_path}")
        loader = UpstageDocumentParseLoader(
            pdf_path, 
            split="page",  # 페이지 단위 분리
            output_format="markdown"
        )
        docs = loader.load()
        
        # 각 문서에 소스 파일명 추가 + 메타데이터 필터링
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.split("/")[-1]
            doc.metadata = filter_complex_metadata(doc.metadata)
        
        all_docs.extend(docs)
        print(f"  → {len(docs)}개 페이지 로드됨")
    
    print(f"\n총 {len(all_docs)}개의 페이지(Parent) 로드됨")
    return all_docs


# =============================================================================
# 3. 헤더 추가 유틸 함수
# =============================================================================
def extract_session_info(source_file: str) -> tuple:
    """
    파일명에서 회기/차수/날짜 정보 추출
    예: '국회본회의_회의록._제429회(14차)_2025년_12월_2일.pdf' 
        -> ('429', '14', '2025년 12월 2일')
    """
    # 회기/차수 추출
    session_pattern = r"제(\d+)회\((\d+)차\)"
    session_match = re.search(session_pattern, source_file)
    session = session_match.group(1) if session_match else "?"
    meeting = session_match.group(2) if session_match else "?"
    
    # 날짜 추출: 2025년_12월_2일 형식
    date_pattern = r"(\d+)년_(\d+)월_(\d+)일"
    date_match = re.search(date_pattern, source_file)
    if date_match:
        date = f"{date_match.group(1)}년 {date_match.group(2)}월 {date_match.group(3)}일"
    else:
        date = ""
    
    return session, meeting, date


def add_context_header(text: str, metadata: dict) -> str:
    """
    Child 청크에 간결한 문맥 헤더 추가
    형식: [429회/14차/2025년 12월 2일]
    """
    source_file = metadata.get("source_file", "")
    session, meeting, date = extract_session_info(source_file)
    
    if date:
        header = f"[{session}회/{meeting}차/{date}] "
    else:
        header = f"[{session}회/{meeting}차] "
    
    return header + text


# =============================================================================
# 4. 커스텀 Child Splitter (헤더 추가 기능 포함)
# =============================================================================
class ContextualChildSplitter(RecursiveCharacterTextSplitter):
    """
    Child 청크에 문맥 헤더를 자동으로 추가하는 커스텀 Splitter
    """
    
    def split_documents(self, documents):
        """문서를 분할하고 각 청크에 헤더 추가"""
        children = super().split_documents(documents)
        
        # 각 Child에 헤더 추가
        for child in children:
            child.page_content = add_context_header(
                child.page_content, 
                child.metadata
            )
        
        return children


# =============================================================================
# 5. Parent-Child Retriever 설정
# =============================================================================
def setup_parent_child_retriever(parent_docs: list):
    """
    LangChain ParentDocumentRetriever를 사용하여 설정합니다.
    Child 청크에 문맥 헤더([429회/14차/날짜])를 자동으로 추가합니다.
    """
    
    # 커스텀 Child Splitter (헤더 추가 기능 포함)
    child_splitter = ContextualChildSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n"]
    )
    
    # Embedding 모델 설정
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")
    
    # Vector Store 설정 (Child 검색용)
    vectorstore = Chroma(
        collection_name="parliament_child_chunks",
        embedding_function=embeddings,
        persist_directory="/Users/parkjehyeong/jungjung/backend/vector_store/child_store"
    )
    
    # Doc Store 설정 (Parent 저장용) - 영구 저장
    # LocalFileStore는 bytes만 저장 가능하므로, Document를 직렬화하는 래퍼 사용
    from langchain_classic.storage import create_kv_docstore
    
    parent_store_path = "/Users/parkjehyeong/jungjung/backend/vector_store/parent_store"
    os.makedirs(parent_store_path, exist_ok=True)
    base_store = LocalFileStore(parent_store_path)
    docstore = create_kv_docstore(base_store)
    
    # ParentDocumentRetriever 생성
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        # parent_splitter는 설정하지 않음 -> parent_docs가 이미 페이지 단위
    )
    
    # 문서 추가 (배치 단위로 분할하여 ChromaDB 제한 우회)
    print("문서 인덱싱 중...")
    batch_size = 100  # 한 번에 100개 Parent씩 처리
    total = len(parent_docs)
    
    for i in range(0, total, batch_size):
        batch = parent_docs[i:i+batch_size]
        retriever.add_documents(batch)
        print(f"  → {min(i+batch_size, total)}/{total} 문서 처리 완료")
    
    print(f"인덱싱 완료: {total}개 Parent 처리됨")
    
    return retriever


# =============================================================================
# 6. 검색 테스트
# =============================================================================
def test_retrieval(retriever, query: str, k: int = 3):
    """
    검색 테스트: Child를 검색하고, 해당 Parent를 반환
    """
    print(f"\n쿼리: {query}")
    print("-" * 50)
    
    results = retriever.invoke(query)
    
    print(f"검색 결과: {len(results)}개의 Parent 문서 반환")
    for i, doc in enumerate(results[:k]):
        print(f"\n[결과 {i+1}]")
        print(f"  메타데이터: {doc.metadata}")
        print(f"  내용 (앞 300자): {doc.page_content[:300]}...")
    
    return results


# =============================================================================
# 7. 메인 실행
# =============================================================================
if __name__ == "__main__":
    # PDF 디렉토리 경로
    pdf_dir = "/Users/parkjehyeong/jungjung/backend/documents/pdf"
    
    # 1. 모든 PDF를 페이지 단위로 로드 (각 페이지 = Parent)
    parent_docs = load_all_pdfs_as_pages(pdf_dir)
    
    # 2. Parent-Child Retriever 설정
    retriever = setup_parent_child_retriever(parent_docs)
    
    # 3. 검색 테스트
    test_retrieval(retriever, "한지아 의원이 대표발의한 법안은?")
    test_retrieval(retriever, "12월 2일 회의")
    test_retrieval(retriever, "검역법 일부개정법률안")
