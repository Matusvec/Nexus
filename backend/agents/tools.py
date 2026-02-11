"""
Tool registry and built-in tool implementations for Nexus agents.

Tools follow a simple interface:
- name: unique identifier
- description: what the tool does (shown to LLM)
- parameters: list of parameter definitions
- execute(**kwargs) -> ToolResult

Built-in tools:
- rag_search: Search the T-Retriever RAG hierarchy
- rag_tree_search: Search specific layers of the RAG tree
- web_search: Search the web (via API)
- youtube_search: Search YouTube for relevant content
- document_list: List available documents
- document_summary: Get document summary
- text_summarize: Summarize long text
- calculate: Evaluate mathematical expressions
- extract_entities: Extract key entities from text
"""

from __future__ import annotations

import json
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable


@dataclass
class ToolResult:
    """Result returned by a tool execution."""
    output: str
    success: bool = True
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "integer", "boolean", "float"
    description: str
    required: bool = True
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }


class Tool:
    """
    A tool that an agent can invoke.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        fn: Callable[..., ToolResult],
        category: str = "general",
    ):
        self.name = name
        self.description = description
        self.parameters = [p.to_dict() for p in parameters]
        self._parameters_raw = parameters
        self._fn = fn
        self.category = category

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        # Apply defaults for missing optional params
        for p in self._parameters_raw:
            if p.name not in kwargs and not p.required and p.default is not None:
                kwargs[p.name] = p.default
        return self._fn(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
        }


class ToolRegistry:
    """
    Central registry for all available tools.
    Agents reference tools by name from this registry.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                output=f"Error: Tool '{name}' not found. Available: {list(self._tools.keys())}",
                success=False,
            )
        try:
            return tool.execute(**args)
        except Exception as e:
            return ToolResult(output=f"Error executing {name}: {str(e)}", success=False)

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_tool_names(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize all tools."""
        return [t.to_dict() for t in self._tools.values()]


# ============================================================================
# BUILT-IN TOOL IMPLEMENTATIONS
# ============================================================================


def _rag_search(query: str, top_k: int = 5, document_id: str = "") -> ToolResult:
    """Search the RAG knowledge base using T-Retriever."""
    try:
        from t_query import collapsed_tree_retrieval
        from storage import get_collection_stats

        stats = get_collection_stats()
        if stats["total_chunks"] == 0:
            return ToolResult(
                output="Knowledge base is empty. No documents have been uploaded yet.",
                success=True,
            )

        results = collapsed_tree_retrieval(
            query=query,
            document_id=document_id if document_id else None,
            top_k=int(top_k),
        )

        if not results:
            return ToolResult(output="No relevant results found.", success=True)

        sources = []
        output_lines = []
        for i, r in enumerate(results[:int(top_k)], 1):
            chunk_id = r.get("id", "unknown")
            text = r.get("text", r.get("document", ""))[:500]
            layer = r.get("layer", 0)
            score = r.get("score", 0)
            doc_id = r.get("document_id", "")

            output_lines.append(
                f"[{i}] (Layer {layer}, Score: {score:.3f}) {text}"
            )
            sources.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "layer": layer,
                    "score": score,
                    "preview": text[:200],
                }
            )

        return ToolResult(
            output="\n\n".join(output_lines), success=True, sources=sources
        )
    except Exception as e:
        return ToolResult(output=f"RAG search error: {str(e)}", success=False)


def _rag_tree_search(
    query: str, layer: int = 0, top_k: int = 5, document_id: str = ""
) -> ToolResult:
    """Search a specific layer of the RAG tree hierarchy."""
    try:
        from storage import get_or_create_collection
        from embeddings import get_embedding

        collection = get_or_create_collection()
        query_emb = get_embedding(query)

        where_filter: Dict[str, Any] = {"layer": int(layer)}
        if document_id:
            where_filter = {
                "$and": [
                    {"layer": int(layer)},
                    {"document_id": document_id},
                ]
            }

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=int(top_k),
            where=where_filter,
        )

        if not results["ids"][0]:
            return ToolResult(
                output=f"No results found at layer {layer}.", success=True
            )

        lines = []
        sources = []
        for i, (cid, doc, meta) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0]),
            1,
        ):
            preview = doc[:500]
            lines.append(f"[{i}] {cid}: {preview}")
            sources.append(
                {
                    "chunk_id": cid,
                    "document_id": meta.get("document_id", ""),
                    "layer": meta.get("layer", 0),
                    "preview": preview[:200],
                }
            )

        return ToolResult(
            output="\n\n".join(lines), success=True, sources=sources
        )
    except Exception as e:
        return ToolResult(
            output=f"Tree search error: {str(e)}", success=False
        )


def _document_list() -> ToolResult:
    """List all documents in the knowledge base."""
    try:
        from storage import get_collection_stats, get_or_create_collection

        stats = get_collection_stats()
        if not stats["documents"]:
            return ToolResult(output="No documents in the knowledge base.", success=True)

        collection = get_or_create_collection()
        lines = []
        for doc in stats["documents"]:
            doc_results = collection.get(where={"document_id": doc})
            lines.append(f"- {doc} ({len(doc_results['ids'])} chunks)")

        return ToolResult(
            output=f"Documents ({len(stats['documents'])}):\n" + "\n".join(lines),
            success=True,
        )
    except Exception as e:
        return ToolResult(output=f"Error listing documents: {str(e)}", success=False)


def _document_summary(document_id: str) -> ToolResult:
    """Get the summary and stats for a specific document."""
    try:
        from storage import get_or_create_collection

        collection = get_or_create_collection()
        results = collection.get(
            where={"document_id": document_id},
            include=["metadatas", "documents"],
        )

        if not results["ids"]:
            return ToolResult(
                output=f"Document '{document_id}' not found.", success=False
            )

        # Get summary from first chunk metadata
        summary = ""
        layers: Dict[int, int] = {}
        for meta in results["metadatas"]:
            if meta.get("doc_summary"):
                summary = meta["doc_summary"]
            layer = meta.get("layer", 0)
            layers[layer] = layers.get(layer, 0) + 1

        layer_info = ", ".join(
            f"Layer {k}: {v} chunks" for k, v in sorted(layers.items())
        )

        output = f"Document: {document_id}\nChunks: {len(results['ids'])}\nLayers: {layer_info}"
        if summary:
            output += f"\nSummary: {summary}"

        return ToolResult(output=output, success=True)
    except Exception as e:
        return ToolResult(
            output=f"Error getting summary: {str(e)}", success=False
        )


def _web_search(query: str, num_results: int = 5) -> ToolResult:
    """
    Search the web for information.
    Uses a simple approach — in production, integrate a real search API.
    """
    try:
        import urllib.request
        import urllib.parse

        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "NexusAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Parse results from DuckDuckGo HTML
        results = []
        snippets = re.findall(
            r'class="result__snippet">(.*?)</a>', html, re.DOTALL
        )
        titles = re.findall(
            r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL
        )
        links = re.findall(
            r'class="result__url"[^>]*href="([^"]*)"', html
        )

        for i in range(min(int(num_results), len(snippets))):
            title = re.sub(r"<[^>]+>", "", titles[i] if i < len(titles) else "")
            snippet = re.sub(r"<[^>]+>", "", snippets[i])
            link = links[i] if i < len(links) else ""
            results.append(f"[{i+1}] {title.strip()}\n    {snippet.strip()}\n    URL: {link}")

        if results:
            return ToolResult(
                output="\n\n".join(results),
                success=True,
                metadata={"query": query, "count": len(results)},
            )
        return ToolResult(
            output=f"No web results found for: {query}",
            success=True,
        )
    except Exception as e:
        return ToolResult(
            output=f"Web search unavailable: {str(e)}. Provide answer from your knowledge.",
            success=False,
        )


def _youtube_search(query: str, num_results: int = 5) -> ToolResult:
    """Search YouTube for relevant videos."""
    try:
        import urllib.request
        import urllib.parse

        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.youtube.com/results?search_query={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "NexusAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Extract video IDs and titles from YouTube search
        video_ids = re.findall(r'"videoId":"([^"]{11})"', html)
        titles_raw = re.findall(r'"title":\{"runs":\[\{"text":"([^"]+)"', html)

        seen = set()
        results = []
        for vid, title in zip(video_ids, titles_raw):
            if vid not in seen and len(results) < int(num_results):
                seen.add(vid)
                results.append(
                    f"[{len(results)+1}] {title}\n    https://youtube.com/watch?v={vid}"
                )

        if results:
            return ToolResult(
                output="\n\n".join(results),
                success=True,
                metadata={"query": query, "count": len(results)},
            )
        return ToolResult(
            output=f"No YouTube results found for: {query}",
            success=True,
        )
    except Exception as e:
        return ToolResult(
            output=f"YouTube search unavailable: {str(e)}. Provide answer from your knowledge.",
            success=False,
        )


def _text_summarize(text: str, max_length: int = 200) -> ToolResult:
    """Summarize a piece of text using the LLM."""
    try:
        from gemini_client import generate_content

        prompt = f"Summarize the following text in {max_length} words or fewer:\n\n{text[:4000]}"
        summary = generate_content(prompt)
        return ToolResult(output=summary, success=True)
    except Exception as e:
        return ToolResult(output=f"Summarization error: {str(e)}", success=False)


def _calculate(expression: str) -> ToolResult:
    """Safely evaluate a mathematical expression using ast parsing."""
    import ast
    import operator

    # Supported binary operations
    _ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Supported math functions
    _funcs = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "log2": math.log2, "exp": math.exp, "ceil": math.ceil,
        "floor": math.floor, "pi": math.pi, "e": math.e,
    }

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in _funcs:
            val = _funcs[node.id]
            if isinstance(val, (int, float)):
                return val
            raise ValueError(f"'{node.id}' is a function, not a value")
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ops:
            return _ops[type(node.op)](_eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _ops:
            return _ops[type(node.op)](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _funcs:
                fn = _funcs[node.func.id]
                if callable(fn):
                    args = [_eval_node(a) for a in node.args]
                    return fn(*args)
            raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        return ToolResult(output=f"{expression} = {result}", success=True)
    except Exception as e:
        return ToolResult(
            output=f"Calculation error for '{expression}': {str(e)}", success=False
        )


def _extract_entities(text: str) -> ToolResult:
    """Extract key entities (people, places, concepts) from text."""
    try:
        from gemini_client import generate_content

        prompt = f"""Extract the key entities from this text. Categorize them as:
- People
- Organizations
- Concepts/Topics
- Locations
- Technical Terms

Return as a structured list.

Text: {text[:3000]}"""
        entities = generate_content(prompt)
        return ToolResult(output=entities, success=True)
    except Exception as e:
        return ToolResult(
            output=f"Entity extraction error: {str(e)}", success=False
        )


# ============================================================================
# TOOL REGISTRATION
# ============================================================================


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry populated with all built-in tools."""
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="rag_search",
            description="Search the RAG knowledge base for relevant documents and chunks. Use this for any question about uploaded documents.",
            parameters=[
                ToolParameter("query", "string", "Search query"),
                ToolParameter("top_k", "integer", "Number of results (default 5)", required=False, default=5),
                ToolParameter("document_id", "string", "Filter to specific document", required=False, default=""),
            ],
            fn=_rag_search,
            category="knowledge",
        )
    )

    registry.register(
        Tool(
            name="rag_tree_search",
            description="Search a specific layer of the RAG tree. Layer 0 = detailed chunks, higher layers = summaries. Useful for getting both detailed and high-level information.",
            parameters=[
                ToolParameter("query", "string", "Search query"),
                ToolParameter("layer", "integer", "Tree layer to search (0=base, 1+=summaries)", required=False, default=0),
                ToolParameter("top_k", "integer", "Number of results", required=False, default=5),
                ToolParameter("document_id", "string", "Filter to specific document", required=False, default=""),
            ],
            fn=_rag_tree_search,
            category="knowledge",
        )
    )

    registry.register(
        Tool(
            name="document_list",
            description="List all documents currently in the knowledge base with chunk counts.",
            parameters=[],
            fn=_document_list,
            category="knowledge",
        )
    )

    registry.register(
        Tool(
            name="document_summary",
            description="Get summary and statistics for a specific document.",
            parameters=[
                ToolParameter("document_id", "string", "Document identifier"),
            ],
            fn=_document_summary,
            category="knowledge",
        )
    )

    registry.register(
        Tool(
            name="web_search",
            description="Search the web for current information, articles, and references.",
            parameters=[
                ToolParameter("query", "string", "Search query"),
                ToolParameter("num_results", "integer", "Number of results", required=False, default=5),
            ],
            fn=_web_search,
            category="web",
        )
    )

    registry.register(
        Tool(
            name="youtube_search",
            description="Search YouTube for tutorial videos, lectures, and educational content.",
            parameters=[
                ToolParameter("query", "string", "Search query"),
                ToolParameter("num_results", "integer", "Number of results", required=False, default=5),
            ],
            fn=_youtube_search,
            category="web",
        )
    )

    registry.register(
        Tool(
            name="text_summarize",
            description="Summarize a piece of text into a shorter version.",
            parameters=[
                ToolParameter("text", "string", "Text to summarize"),
                ToolParameter("max_length", "integer", "Max words in summary", required=False, default=200),
            ],
            fn=_text_summarize,
            category="analysis",
        )
    )

    registry.register(
        Tool(
            name="calculate",
            description="Evaluate a mathematical expression. Supports standard math functions (sin, cos, sqrt, log, etc.).",
            parameters=[
                ToolParameter("expression", "string", "Math expression to evaluate"),
            ],
            fn=_calculate,
            category="utility",
        )
    )

    registry.register(
        Tool(
            name="extract_entities",
            description="Extract key entities (people, organizations, concepts, technical terms) from text.",
            parameters=[
                ToolParameter("text", "string", "Text to analyze"),
            ],
            fn=_extract_entities,
            category="analysis",
        )
    )

    return registry
