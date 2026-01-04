"""
HyDE (Hypothetical Document Embedding) 생성기
=============================================
쿼리를 받아 가상의 문서를 생성하고, 
그 문서로 검색하여 더 정확한 결과를 얻습니다.

흐름:
    사용자 쿼리
         │
         ▼
    LLM (solar-pro2) → 가상 문서 생성
         │
         ▼
    앙상블 리트리버로 검색
         │
         ▼
    결과 반환
"""

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 대한민국 국회 회의록 검색을 돕는 전문가입니다.
사용자의 질문을 받아, 해당 질문에 대한 답변이 포함되어 있을 법한 
국회 회의록의 일부를 가상으로 작성해주세요.

작성 규칙:
1. 실제 국회 회의록처럼 작성하세요 (발언자, 날짜 등 포함 가능)
2. 질문의 핵심 키워드와 관련 용어를 자연스럽게 포함하세요
3. 150-300자 내외로 간결하게 작성하세요
4. 실제 사실이 아니어도 됩니다. 검색용 가상 문서입니다."""),
    ("human", "질문: {query}\n\n위 질문에 대한 답변이 포함된 가상의 국회 회의록 일부를 작성해주세요.")
])


class HyDEGenerator:
    """HyDE 가상 문서 생성기"""
    
    def __init__(self, model_name: str = "solar-pro2"):
        self.llm = ChatUpstage(model=model_name)
        self.chain = HYDE_PROMPT | self.llm | StrOutputParser()
    
    def generate(self, query: str) -> str:
        """
        쿼리를 받아 가상의 문서를 생성
        
        Args:
            query: 사용자 검색 쿼리
        
        Returns:
            가상 문서 텍스트
        """
        hypothetical_doc = self.chain.invoke({"query": query})
        return hypothetical_doc
    
    def generate_with_original(self, query: str) -> str:
        """
        가상 문서 + 원본 쿼리를 결합 (검색 정확도 향상)
        """
        hypothetical_doc = self.generate(query)
        return f"{query}\n\n{hypothetical_doc}"
