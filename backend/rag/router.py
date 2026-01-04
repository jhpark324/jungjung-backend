"""
질문 라우터 (Router)
===================
사용자 질문을 분류하여 적절한 에이전트로 라우팅합니다.

분류:
- 국회회의록 관련 → RAG 에이전트
- 그 외 → 일반 LLM 응답
"""

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 질문 분류 전문가입니다.
사용자의 질문이 "대한민국 국회 회의록"과 관련된 질문인지 판단하세요.

국회 회의록 관련 질문 예시:
- 특정 의원의 발언, 발의 법안
- 국회 본회의, 위원회 회의 내용
- 법률안, 의안 관련 질문
- 국회 일정, 회의 날짜
- 특정 정책에 대한 국회 논의

국회 회의록과 관련 없는 질문 예시:
- 일반 상식 질문
- 날씨, 시간 등 일상 질문
- 다른 기관/조직 관련 질문
- 개인적인 조언 요청

반드시 "RAG" 또는 "LLM" 중 하나만 답변하세요."""),
    ("human", "질문: {query}\n\n이 질문은 국회 회의록 관련 질문인가요? (RAG 또는 LLM으로만 답변)")
])


class QuestionRouter:
    """질문 라우터"""
    
    def __init__(self, model_name: str = "solar-pro2"):
        self.llm = ChatUpstage(model=model_name)
        self.chain = ROUTER_PROMPT | self.llm | StrOutputParser()
    
    def route(self, query: str) -> str:
        """
        질문을 분류하여 라우팅 결정
        
        Args:
            query: 사용자 질문
        
        Returns:
            "RAG" 또는 "LLM"
        """
        result = self.chain.invoke({"query": query})
        
        # 결과 정규화
        result = result.strip().upper()
        if "RAG" in result:
            return "RAG"
        else:
            return "LLM"
