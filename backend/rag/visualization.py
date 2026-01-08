"""
Evidence Visualization - 근거 문서 시각화 모듈
==============================================
RAG에서 검색된 문서를 분석하여 적합한 시각화를 생성합니다.

지원 그래프:
    - tfidf_keywords: TF-IDF 핵심 단어 막대 그래프
    - cooccurrence_heatmap: 키워드 동시출현 히트맵
    - word_cloud: 워드클라우드
    - ngram_frequency: N-gram 빈도 그래프
    - topic_treemap: 주제 분포 파이 차트
"""

import os
import re
import json
import math
from typing import Dict, List, Any, Optional
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from langchain_upstage import ChatUpstage
from langchain_core.documents import Document


# =============================================================================
# 설정
# =============================================================================
# Mac용 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리 (backend/static/graphs)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 그래프 템플릿 정의
# =============================================================================
GRAPH_TEMPLATES = {
    "tfidf_keywords": {
        "name": "TF-IDF 핵심 단어",
        "requires_llm": False,
    },
    "cooccurrence_heatmap": {
        "name": "키워드 동시 출현 히트맵",
        "requires_llm": False,
    },
    "word_cloud": {
        "name": "워드 클라우드",
        "requires_llm": False,
    },
    "ngram_frequency": {
        "name": "N-gram 빈도 그래프",
        "requires_llm": False,
    },
    "topic_treemap": {
        "name": "주제 분포 트리맵",
        "requires_llm": True,
    }
}


# =============================================================================
# 텍스트 전처리
# =============================================================================
def clean_text(text: str) -> str:
    """텍스트 전처리: 메타데이터 태그, 특수문자 제거"""
    # 메타데이터 태그 제거
    text = re.sub(r'\[문서 정보\][^-]*---', '', text)
    text = re.sub(r'\[메타데이터\][^\-]*---', '', text)
    text = re.sub(r'\[주제\d*:[^\]]*\]', '', text)
    text = re.sub(r'회의:[^-]*-', '', text)
    text = re.sub(r'날짜:[^-]*-', '', text)
    text = re.sub(r'유형:[^-]*-', '', text)
    text = re.sub(r'섹션:[^-]*-', '', text)
    text = re.sub(r'표결결과[^-]*---', '', text)
    text = re.sub(r'문서명:[^\n]*\n', '', text)
    text = re.sub(r'회기:[^\n]*\n', '', text)
    text = re.sub(r'차수:[^\n]*\n', '', text)
    # 특수문자 및 불필요한 공백 정리
    text = re.sub(r'---+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def documents_to_text(documents: List[Document]) -> str:
    """Document 리스트를 텍스트로 변환"""
    chunks = [doc.page_content for doc in documents]
    cleaned = [clean_text(chunk) for chunk in chunks]
    return " ".join(cleaned)


# =============================================================================
# 규칙 기반 분석
# =============================================================================
# 확장된 불용어
STOPWORDS = {
    # 조사/어미
    '의', '가', '이', '을', '를', '에', '에서', '로', '으로', '는', '은', '과', '와',
    '도', '만', '등', '및', '한', '할', '하는', '된', '수', '것', '있는', '있습니다',
    '합니다', '대한', '일부', '있어', '대표', '발의', '입니다', '됩니다', '습니다',
    # 메타데이터 관련
    '메타데이터', '회의', '날짜', '유형', '섹션', '표결', '결과', '문서', '정보',
    # 일반 표현
    '그래서', '따라', '이번', '현재', '경우', '위해', '통해', '관한', '관련',
    '기준', '규정', '해지', '조건', '상정', '가지', '때문', '또한', '하여',
    # 의회/토론 관련
    '존경하는', '여러분', '의원', '위원', '위원회', '국회', '국회의원', '선포',
    '말씀', '드립니다', '드리겠습니다', '감사합니다', '동의', '안건', '반대', '찬성',
    # 추가 불용어
    '그것', '이것', '저것', '우리', '자신', '매우', '정말', '아주', '너무',
    '처럼', '같이', '어떤', '모든', '어디', '언제', '얼마', '누구', '왜',
    '어떻게', '무엇', '그리고', '그러나', '하지만', '그래도', '즉', '곧'
}


def normalize_korean_word(word: str) -> str:
    """한국어 조사 제거"""
    if len(word) < 2:
        return word
    suffixes_2 = ['에서', '으로', '에는', '과는', '와는', '오후', '오전']
    for s in suffixes_2:
        if word.endswith(s) and len(word) > len(s):
            return word[:-len(s)]
    suffixes_1 = ['은', '는', '이', '가', '을', '를', '의', '에', '로', '과', '와', '도', '만']
    if word[-1] in suffixes_1:
        return word[:-1]
    return word


def analyze_text_rule_based(text: str, selected_graphs: List[str]) -> Dict[str, Any]:
    """규칙 기반 텍스트 분석"""
    results = {}
    
    # 단어 추출 및 정규화
    raw_words = re.findall(r'[가-힣]{2,}', text)
    processed_words = []
    for w in raw_words:
        if w in STOPWORDS:
            continue
        norm_w = normalize_korean_word(w)
        if len(norm_w) >= 2 and norm_w not in STOPWORDS:
            processed_words.append(norm_w)
    
    word_counts = Counter(processed_words)
    
    # TF-IDF 계산
    if "tfidf_keywords" in selected_graphs:
        total_words = len(processed_words) if processed_words else 1
        tfidf_scores = {}
        for word, count in word_counts.most_common(15):
            tf = count / total_words
            idf = math.log(total_words / (count + 1)) + 1
            tfidf_scores[word] = round(tf * idf, 4)
        sorted_tfidf = dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10])
        results["tfidf_keywords"] = sorted_tfidf
    
    # 동시출현 히트맵
    if "cooccurrence_heatmap" in selected_graphs:
        sentences = re.split(r'[.!?。]', text)
        top_words = [w for w, _ in word_counts.most_common(8)]
        cooc_matrix = [[0] * len(top_words) for _ in range(len(top_words))]
        
        for sentence in sentences:
            sent_words = set(re.findall(r'[가-힣]{2,}', sentence))
            sent_words = {normalize_korean_word(w) for w in sent_words if w not in STOPWORDS}
            for i, w1 in enumerate(top_words):
                for j, w2 in enumerate(top_words):
                    if w1 in sent_words and w2 in sent_words:
                        cooc_matrix[i][j] += 1
        
        results["cooccurrence_heatmap"] = {
            "labels": top_words,
            "matrix": cooc_matrix
        }
    
    # 워드 클라우드
    if "word_cloud" in selected_graphs:
        results["word_cloud"] = dict(word_counts.most_common(20))
    
    # N-gram 빈도
    if "ngram_frequency" in selected_graphs:
        sentences = re.split(r'[.!?。]', text)
        ngrams = []
        for sentence in sentences:
            words = re.findall(r'[가-힣]+', sentence)
            for i in range(len(words) - 1):
                ngram = ' '.join(words[i:i+2])
                if len(ngram) >= 4:
                    ngrams.append(ngram)
        results["ngram_frequency"] = dict(Counter(ngrams).most_common(8))
    
    return results


# =============================================================================
# 그래프 렌더링 함수들
# =============================================================================
def render_tfidf_keywords(data: Dict[str, float], query_id: str) -> Optional[str]:
    """TF-IDF 기반 핵심 단어 막대 그래프"""
    if not data:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    words = list(data.keys())[:10]
    scores = [data[w] for w in words]
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(words)))
    bars = ax.barh(words, scores, color=colors, edgecolor='darkgreen', linewidth=0.5)
    
    ax.set_xlabel('TF-IDF 점수', fontsize=12)
    ax.set_title(f'[TF-IDF 핵심 단어]', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    filename = f"{query_id}_tfidf_keywords.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"/static/graphs/{filename}"


def render_cooccurrence_heatmap(data: Dict[str, Any], query_id: str) -> Optional[str]:
    """키워드 동시 출현 히트맵"""
    if not data or not data.get('matrix'):
        return None
    
    matrix = np.array(data['matrix'])
    labels = data['labels']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > 0:
                text_color = 'white' if matrix[i, j] > matrix.max()/2 else 'black'
                ax.text(j, i, str(int(matrix[i, j])), ha='center', va='center',
                       color=text_color, fontsize=8)
    
    ax.set_title('[키워드 동시출현 히트맵]', fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('동시 출현 횟수', fontsize=10)
    
    plt.tight_layout()
    filename = f"{query_id}_cooccurrence_heatmap.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"/static/graphs/{filename}"


def render_word_cloud(data: Dict[str, int], query_id: str) -> Optional[str]:
    """워드 클라우드 (matplotlib 기반)"""
    if not data:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    words = list(data.keys())[:20]
    counts = [data[w] for w in words]
    max_count = max(counts) if counts else 1
    
    np.random.seed(42)  # 재현성
    for i, (word, count) in enumerate(zip(words, counts)):
        size = 12 + (count / max_count) * 28
        x = 10 + (i % 5) * 18 + np.random.uniform(-3, 3)
        y = 85 - (i // 5) * 18 + np.random.uniform(-3, 3)
        
        color = plt.cm.viridis(count / max_count)
        ax.text(x, y, word, fontsize=size, ha='center', va='center',
               color=color, fontweight='bold', alpha=0.8)
    
    ax.set_title('[워드 클라우드]', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filename = f"{query_id}_word_cloud.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"/static/graphs/{filename}"


def render_ngram_frequency(data: Dict[str, int], query_id: str) -> Optional[str]:
    """N-gram 빈도 막대 그래프"""
    if not data:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ngrams = list(data.keys())[:8]
    counts = [data[n] for n in ngrams]
    
    colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(ngrams)))
    bars = ax.barh(ngrams, counts, color=colors, edgecolor='purple', linewidth=0.5)
    
    ax.set_xlabel('빈도', fontsize=12)
    ax.set_title('[N-gram 빈도 분석]', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    filename = f"{query_id}_ngram_frequency.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"/static/graphs/{filename}"


def render_topic_treemap(data: List[Dict], query_id: str) -> Optional[str]:
    """주제 분포 파이 차트"""
    if not data:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    names = [d['name'] for d in data]
    percentages = [d['percentage'] for d in data]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    
    wedges, texts, autotexts = ax.pie(
        percentages,
        labels=names,
        autopct='%1.0f%%',
        colors=colors,
        explode=[0.02] * len(names),
        shadow=True,
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax.set_title('[주제 분포]', fontsize=14, fontweight='bold')
    
    legend_labels = [f"{d['name']}: {', '.join(d.get('keywords', [])[:2])}" for d in data]
    ax.legend(wedges, legend_labels, title="주요 키워드", loc="lower center",
              bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
    
    plt.tight_layout()
    filename = f"{query_id}_topic_treemap.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"/static/graphs/{filename}"


# =============================================================================
# LLM 기반 그래프 선택
# =============================================================================
GRAPH_SELECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "graph_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "selected_graphs": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["tfidf_keywords", "cooccurrence_heatmap", "word_cloud", 
                                "ngram_frequency", "topic_treemap"]
                    }
                },
                "selection_reason": {"type": "string"},
                "topic_analysis": {
                    "type": "object",
                    "properties": {
                        "topics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "percentage": {"type": "integer"},
                                    "keywords": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["name", "percentage", "keywords"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["topics"],
                    "additionalProperties": False
                }
            },
            "required": ["selected_graphs", "selection_reason", "topic_analysis"],
            "additionalProperties": False
        }
    }
}


class GraphSelector:
    """LLM 기반 그래프 선택기"""
    
    def __init__(self, model_name: str = "solar-pro2"):
        self.llm = ChatUpstage(model=model_name)
    
    def select_graphs(self, text: str) -> Dict[str, Any]:
        """
        텍스트를 분석하여 적합한 그래프 2개 선택
        
        Returns:
            {
                "selected_graphs": List[str],
                "selection_reason": str,
                "topic_analysis": {"topics": [...]}
            }
        """
        if not text or len(text) < 50:
            return {
                "selected_graphs": ["tfidf_keywords", "ngram_frequency"],
                "selection_reason": "텍스트가 너무 짧아 기본 그래프 선택",
                "topic_analysis": {"topics": []}
            }
        
        prompt = f"""당신은 RAG 시스템의 근거 자료를 분석하여 적절한 시각화 그래프를 추천하는 전문가입니다.

다음 근거 자료들을 분석하고:
1. 가장 적합한 2개의 그래프 유형을 선택하세요
2. 주제 분석 결과를 제공하세요

[근거 자료]
{text[:3000]}

[선택 가능한 그래프]
- tfidf_keywords: TF-IDF 기반 핵심 단어
- cooccurrence_heatmap: 키워드 동시출현 히트맵
- word_cloud: 워드 클라우드
- ngram_frequency: 연속 어구 빈도
- topic_treemap: 주제 분포

반드시 2개의 그래프를 선택하세요."""

        try:
            response = self.llm.invoke(
                prompt,
                response_format=GRAPH_SELECTION_SCHEMA
            )
            result = json.loads(response.content)
            return result
        except Exception as e:
            # Fallback
            return {
                "selected_graphs": ["tfidf_keywords", "cooccurrence_heatmap"],
                "selection_reason": f"LLM 오류로 기본 그래프 선택: {str(e)}",
                "topic_analysis": {"topics": []}
            }


# =============================================================================
# 통합 시각화 생성기
# =============================================================================
class EvidenceVisualizer:
    """근거 문서 시각화 생성기"""
    
    def __init__(self, model_name: str = "solar-pro2"):
        self.graph_selector = GraphSelector(model_name)
    
    def visualize(
        self, 
        documents: List[Document],
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        문서를 분석하여 시각화 생성
        
        Args:
            documents: RAG에서 검색된 문서 리스트
            query_id: 고유 쿼리 ID (파일명용)
        
        Returns:
            {
                "selected_graphs": List[str],
                "selection_reason": str,
                "rendered_graphs": {"graph_type": "url", ...},
                "topic_analysis": {...},
                "errors": List[str]
            }
        """
        if query_id is None:
            query_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        errors = []
        
        # 1. 문서를 텍스트로 변환
        combined_text = documents_to_text(documents)
        
        if len(combined_text) < 50:
            return {
                "selected_graphs": [],
                "selection_reason": "문서 내용이 부족함",
                "rendered_graphs": {},
                "topic_analysis": {"topics": []},
                "errors": ["Text too short for visualization"]
            }
        
        # 2. LLM으로 그래프 선택
        selection_result = self.graph_selector.select_graphs(combined_text)
        selected_graphs = selection_result.get("selected_graphs", ["tfidf_keywords"])
        topic_analysis = selection_result.get("topic_analysis", {"topics": []})
        
        # 3. 규칙 기반 분석
        rule_results = analyze_text_rule_based(combined_text, selected_graphs)
        
        # 4. 그래프 렌더링
        rendered_graphs = {}
        
        for graph_type in selected_graphs:
            try:
                if graph_type == "tfidf_keywords" and graph_type in rule_results:
                    path = render_tfidf_keywords(rule_results[graph_type], query_id)
                    if path:
                        rendered_graphs[graph_type] = path
                        
                elif graph_type == "cooccurrence_heatmap" and graph_type in rule_results:
                    path = render_cooccurrence_heatmap(rule_results[graph_type], query_id)
                    if path:
                        rendered_graphs[graph_type] = path
                        
                elif graph_type == "word_cloud" and graph_type in rule_results:
                    path = render_word_cloud(rule_results[graph_type], query_id)
                    if path:
                        rendered_graphs[graph_type] = path
                        
                elif graph_type == "ngram_frequency" and graph_type in rule_results:
                    path = render_ngram_frequency(rule_results[graph_type], query_id)
                    if path:
                        rendered_graphs[graph_type] = path
                        
                elif graph_type == "topic_treemap":
                    topics = topic_analysis.get("topics", [])
                    if topics:
                        path = render_topic_treemap(topics, query_id)
                        if path:
                            rendered_graphs[graph_type] = path
                            
            except Exception as e:
                errors.append(f"Render error ({graph_type}): {str(e)}")
        
        return {
            "selected_graphs": selected_graphs,
            "selection_reason": selection_result.get("selection_reason", ""),
            "rendered_graphs": rendered_graphs,
            "topic_analysis": topic_analysis,
            "errors": errors
        }
