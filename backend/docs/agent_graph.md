# Graph Visualization

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	router(router)
	rag_agent(rag_agent)
	llm(llm)
	__end__([<p>__end__</p>]):::last
	__start__ --> router;
	router -. &nbsp;LLM&nbsp; .-> llm;
	router -. &nbsp;RAG&nbsp; .-> rag_agent;
	llm --> __end__;
	rag_agent --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
