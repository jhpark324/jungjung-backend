# 국회 회의록 RAG API 명세서

> **Version**: 4.0.0  
> **Base URL**: `http://localhost:8000`  

---

## 1. 검색 API

### `GET /api/search`

질문을 분석하여 국회 회의록 관련이면 RAG, 아니면 일반 LLM으로 처리합니다.

### 요청 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|:----:|--------|------|
| `q` | string | ✅ | - | 검색 쿼리 |
| `limit` | int | ❌ | 5 | 반환할 문서 수 (1-20) |
| `use_hyde` | bool | ❌ | true | HyDE 사용 여부 |
| `enable_viz` | bool | ❌ | true | 시각화 생성 여부 |

---

### 응답 필드

#### 공통

| 필드 | 타입 | 설명 |
|------|------|------|
| `query` | string | 원본 검색 쿼리 |
| `route` | "RAG" \| "LLM" | 라우팅 결과 |
| `answer` | string | LLM이 생성한 답변 |

#### RAG 전용

| 필드 | 타입 | 설명 |
|------|------|------|
| `hypothetical_doc` | string | HyDE 가상 문서 |
| `source_count` | int | 검색된 문서 수 |
| `sources` | Source[] | 검색된 문서 목록 |
| `visualization` | Visualization | 시각화 결과 |

#### Source 객체

| 필드 | 타입 | 설명 |
|------|------|------|
| `content` | string | 문서 내용 (최대 500자) |
| `metadata` | object | source, page, session, meeting 등 |

#### Visualization 객체

| 필드 | 타입 | 설명 |
|------|------|------|
| `selected_graphs` | string[] | 선택된 그래프 유형 (2개) |
| `selection_reason` | string | 그래프 선택 이유 |
| `rendered_graphs` | object | 그래프 URL 맵 |
| `topic_analysis` | object | 주제 분석 결과 |
| `errors` | string[] | 에러 목록 |

#### 그래프 유형

| 값 | 설명 |
|-----|------|
| `tfidf_keywords` | TF-IDF 키워드 막대 그래프 |
| `cooccurrence_heatmap` | 키워드 동시출현 히트맵 |
| `word_cloud` | 워드 클라우드 |
| `ngram_frequency` | N-gram 빈도 그래프 |
| `topic_treemap` | 주제 분포 파이 차트 |

---

### 전체 응답 구조 (RAG 모드)

```json
{
  "query": "쿠팡의 문제점",
  "route": "RAG",
  "answer": "국회 회의록에 따르면...",
  "hypothetical_doc": "[2025년 12월 2일] 쿠팡 관련...",
  "source_count": 5,
  "sources": [
    {
      "content": "[429회/14차] 본회의 내용...",
      "metadata": { "source": "xxx.pdf", "page": 15 }
    }
  ],
  "visualization": {
    "selected_graphs": ["tfidf_keywords", "cooccurrence_heatmap"],
    "selection_reason": "핵심 키워드 분석이 필요합니다",
    "rendered_graphs": {
      "tfidf_keywords": "/static/graphs/xxx_tfidf.png",
      "cooccurrence_heatmap": "/static/graphs/xxx_heatmap.png"
    },
    "topic_analysis": {
      "topics": [
        { "name": "노동 문제", "percentage": 45, "keywords": ["노조", "해고"] }
      ]
    },
    "errors": []
  }
}
```

### 전체 응답 구조 (LLM 모드)

```json
{
  "query": "오늘 날씨 어때?",
  "route": "LLM",
  "answer": "저는 국회 회의록 검색 시스템입니다..."
}
```

---

## 2. 헬스 체크

### `GET /health`

```json
{ "status": "healthy" }
```

---

## 3. 정적 파일

### `GET /static/graphs/{filename}`

시각화 그래프 이미지 조회

---

## React 예시

```tsx
const API_URL = "http://localhost:8000";

// API 호출
async function search(query: string) {
  const res = await fetch(
    `${API_URL}/api/search?q=${encodeURIComponent(query)}`
  );
  return res.json();
}

// 사용
const result = await search("쿠팡의 문제점");

console.log(result.route);   // "RAG" or "LLM"
console.log(result.answer);  // 답변
console.log(result.sources); // 출처 문서들

// 그래프 이미지 표시
if (result.visualization) {
  const graphs = result.visualization.rendered_graphs;
  // { "tfidf_keywords": "/static/graphs/xxx.png" }
  
  Object.entries(graphs).map(([type, url]) => (
    <img src={`${API_URL}${url}`} alt={type} />
  ));
}
```
