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

from datetime import datetime, timedelta
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 대한민국 국회 회의록 검색을 돕는 전문가입니다.
사용자의 질문을 받아, 해당 질문에 대한 답변이 포함되어 있을 법한 
국회 회의록의 일부를 가상으로 작성해주세요.

## 현재 날짜 정보
- 오늘 날짜: {current_date}

## 중요: 회의록 문서 헤더 형식
우리 데이터베이스의 모든 회의록 청크는 다음과 같은 헤더 형식을 가집니다:

```
[429회/14차/2025년 12월 2일] 본회의가 개의되었습니다...
```

**가상 문서를 작성할 때 반드시 이 [회차/차수/날짜] 헤더 형식을 포함하세요!**

## 중요: 모르는 정보는 생략하세요!
- 회차를 모르면: [2025년 12월 2일] (회차/차수 생략)
- 차수를 모르면: [429회/2025년 12월 2일] (차수 생략)
- 날짜를 모르면: [429회/14차] (날짜 생략)
- **절대 임의의 숫자를 넣지 마세요!**

## 날짜 처리 규칙
사용자 질문에 날짜 관련 표현이 있으면:

1. 상대적 날짜 → 구체적 날짜로 변환:
   - "오늘" → {current_date}
   - "어제" → {yesterday_date}
   - "최근", "가장 최근" → [430회] (가장 높은 회차)

## 가상 문서 작성 예시

질문: "12월 2일 회의록 알려줘"
가상 문서:
```
[2025년 12월 2일] 본회의가 오후 8시 30분에 개의되었습니다. 우원식 의장이 성원이 되었으므로 본회의를 개의하겠습니다라고 선언했습니다.
```

질문: "429회 회의록 알려줘"
가상 문서:
```
[429회] 제429회 국회 본회의 회의록입니다. 의장이 성원이 되었으므로 본회의를 개의하겠습니다라고 선언했습니다.
```

질문: "가장 최근 회의록 알려줘"
가상 문서:
```
[430회] 제430회 국회 본회의 회의록입니다. 가장 최근에 개최된 본회의로서 의장이 회의 개의를 선언하였습니다.
```

질문: "한지아 의원이 발의한 법안은?"
가상 문서:
```
한지아 의원이 대표발의한 검역법 일부개정법률안입니다. 이 법안은 검역 관련 규정을 개정하여...
```

## 작성 규칙
1. **아는 정보만 헤더에 포함하세요. 모르면 생략!**
2. 150-300자 내외로 간결하게 작성하세요
3. 실제 사실이 아니어도 됩니다. 검색용 가상 문서입니다."""),
    ("human", "질문: {query}\n\n위 질문에 대한 답변이 포함된 가상의 국회 회의록 일부를 작성해주세요.")
])


class HyDEGenerator:
    """HyDE 가상 문서 생성기"""
    
    def __init__(self, model_name: str = "solar-pro2"):
        self.llm = ChatUpstage(model=model_name)
        self.chain = HYDE_PROMPT | self.llm | StrOutputParser()
    
    def _get_date_context(self) -> dict:
        """현재 날짜 컨텍스트 생성"""
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        return {
            "current_date": today.strftime("%Y년%m월%d일"),
            "yesterday_date": yesterday.strftime("%Y년%m월%d일"),
            "current_year": str(today.year),
        }
    
    def generate(self, query: str) -> str:
        """
        쿼리를 받아 가상의 문서를 생성
        
        Args:
            query: 사용자 검색 쿼리
        
        Returns:
            가상 문서 텍스트
        """
        # 날짜 컨텍스트 주입
        date_context = self._get_date_context()
        
        hypothetical_doc = self.chain.invoke({
            "query": query,
            **date_context
        })
        return hypothetical_doc
    
    def generate_with_original(self, query: str) -> str:
        """
        가상 문서 + 원본 쿼리를 결합 (검색 정확도 향상)
        """
        hypothetical_doc = self.generate(query)
        return f"{query}\n\n{hypothetical_doc}"

