"""
LangGraph 시각화 스크립트
========================
RAG 그래프와 Agent 그래프를 시각화하여 이미지로 저장합니다.

사용법:
    cd /Users/parkjehyeong/jungjung
    uv run python -m backend.scripts.visualize_graph
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def visualize_rag_graph():
    """RAG 그래프 시각화"""
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Optional, List
    
    class RAGState(TypedDict):
        query: str
        hypothetical_doc: Optional[str]
        search_query: str
        documents: list
        answer: str
    
    graph = StateGraph(RAGState)
    graph.add_node("hyde", lambda x: x)
    graph.add_node("retrieve", lambda x: x)
    graph.add_node("generate", lambda x: x)
    
    graph.add_edge(START, "hyde")
    graph.add_edge("hyde", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()


def visualize_agent_graph():
    """Agent 그래프 시각화"""
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Optional, Literal
    
    class AgentState(TypedDict):
        query: str
        route: Literal["RAG", "LLM"]
        answer: str
    
    graph = StateGraph(AgentState)
    graph.add_node("router", lambda x: x)
    graph.add_node("rag_agent", lambda x: x)
    graph.add_node("llm", lambda x: x)
    
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda x: x.get("route", "RAG"),
        {"RAG": "rag_agent", "LLM": "llm"}
    )
    graph.add_edge("rag_agent", END)
    graph.add_edge("llm", END)
    
    return graph.compile()


def save_graph_image(compiled_graph, filename: str, output_dir: Path):
    """그래프를 PNG 이미지로 저장"""
    try:
        # PNG 이미지 생성 (graphviz 필요)
        png_data = compiled_graph.get_graph().draw_mermaid_png()
        
        filepath = output_dir / filename
        with open(filepath, "wb") as f:
            f.write(png_data)
        
        print(f"✅ 저장 완료: {filepath}")
        return filepath
    except Exception as e:
        print(f"⚠️ PNG 생성 실패: {e}")
        print("   Mermaid 다이어그램으로 대체합니다.")
        return None


def save_mermaid_diagram(compiled_graph, filename: str, output_dir: Path):
    """Mermaid 다이어그램을 Markdown 파일로 저장"""
    mermaid_code = compiled_graph.get_graph().draw_mermaid()
    
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        f.write("# Graph Visualization\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```\n")
    
    print(f"✅ Mermaid 저장 완료: {filepath}")
    return filepath


def main():
    # 출력 디렉토리 설정
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)
    
    print("🔗 LangGraph 시각화 생성 중...\n")
    
    # RAG 그래프
    print("=== RAG Graph ===")
    rag_graph = visualize_rag_graph()
    save_graph_image(rag_graph, "rag_graph.png", output_dir)
    save_mermaid_diagram(rag_graph, "rag_graph.md", output_dir)
    
    print()
    
    # Agent 그래프
    print("=== Agent Graph ===")
    agent_graph = visualize_agent_graph()
    save_graph_image(agent_graph, "agent_graph.png", output_dir)
    save_mermaid_diagram(agent_graph, "agent_graph.md", output_dir)
    
    print("\n" + "="*50)
    print(f"📁 출력 디렉토리: {output_dir}")
    print("="*50)
    
    # Mermaid 다이어그램 출력
    print("\n### RAG Graph Mermaid ###")
    print(rag_graph.get_graph().draw_mermaid())
    
    print("\n### Agent Graph Mermaid ###")
    print(agent_graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
