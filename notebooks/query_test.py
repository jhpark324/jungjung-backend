"""
VectorStore 쿼리 테스트
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

# Embedding 모델
embeddings = UpstageEmbeddings(model="embedding-query")

# 기존 VectorStore 로드
vectorstore = Chroma(
    collection_name="parliament_child_chunks",
    embedding_function=embeddings,
    persist_directory="/Users/parkjehyeong/jungjung/backend/vector_store/child_store"
)

# 컬렉션 정보 확인
collection = vectorstore._collection
print(f"저장된 문서 수: {collection.count()}")

if collection.count() > 0:
    # 쿼리 테스트
    queries = [
        "한지아 의원이 대표발의한 법안은?",
        "12월 2일 회의",
        "검역법"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"쿼리: {query}")
        print(f"{'='*50}")
        
        results = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n[결과 {i}]")
            print(f"  내용 (앞 300자): {doc.page_content[:300]}...")
            print(f"  메타데이터: {doc.metadata}")
else:
    print("저장된 문서가 없습니다.")
