"""
답변 생성기 (Generator)
======================
검색된 문서를 기반으로 사용자 질문에 대한 답변을 생성합니다.
"""

from typing import List
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 대한민국 국회 회의록을 분석하는 전문 비서입니다.
주어진 회의록 내용을 바탕으로 사용자의 질문에 정확하고 친절하게 답변하세요.

답변 규칙:
1. 제공된 회의록 내용만을 기반으로 답변하세요.
2. 회의록에 없는 내용은 "제공된 회의록에서 해당 내용을 찾을 수 없습니다"라고 답변하세요.
3. 가능하면 발언자, 날짜, 회의명 등 구체적인 정보를 포함하세요.
4. 답변은 명확하고 이해하기 쉽게 작성하세요.
5. 필요한 경우 회의록 내용을 인용하세요."""),
    ("human", """다음은 검색된 국회 회의록입니다:

{context}

---

질문: {question}

위 회의록 내용을 바탕으로 질문에 답변해주세요.""")
])


class AnswerGenerator:
    """RAG 답변 생성기"""
    
    def __init__(self, model_name: str = "solar-pro2"):
        self.llm = ChatUpstage(model=model_name)
        self.chain = GENERATE_PROMPT | self.llm | StrOutputParser()
    
    def generate(self, question: str, documents: List[Document]) -> str:
        """
        검색된 문서를 기반으로 답변 생성
        
        Args:
            question: 사용자 질문
            documents: 검색된 문서 리스트
        
        Returns:
            생성된 답변
        """
        # 문서 내용을 컨텍스트로 변환
        context = self._format_documents(documents)
        
        # 답변 생성
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        return answer
    
    def _format_documents(self, documents: List[Document]) -> str:
        """문서 리스트를 컨텍스트 문자열로 변환"""
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source = metadata.get("source_file", "알 수 없음")
            page = metadata.get("page", "알 수 없음")
            
            formatted.append(f"[문서 {i}] (출처: {source}, 페이지: {page})")
            formatted.append(doc.page_content)
            formatted.append("")  # 빈 줄
        
        return "\n".join(formatted)
