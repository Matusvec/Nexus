"""
Adapters package — integration stubs for upstream systems.

Each adapter provides a stable interface that the agentic framework uses.
When the upstream module is absent (pre-merge), adapters fall back to
mock/dev implementations so the system can still run.
"""

from agents.adapters.retrieval_adapter import (
    query as retrieval_query,
    explain as retrieval_explain,
    list_documents as retrieval_list_documents,
    get_document_summary as retrieval_get_document_summary,
    get_stats as retrieval_get_stats,
    MOCK_MODE as RETRIEVAL_MOCK_MODE,
)

from agents.adapters.hitl_adapter import (
    request_approval as hitl_request_approval,
    report_tool_result as hitl_report_tool_result,
    ACTIVE_MODE as HITL_ACTIVE_MODE,
)

__all__ = [
    "retrieval_query",
    "retrieval_explain",
    "retrieval_list_documents",
    "retrieval_get_document_summary",
    "retrieval_get_stats",
    "RETRIEVAL_MOCK_MODE",
    "hitl_request_approval",
    "hitl_report_tool_result",
    "HITL_ACTIVE_MODE",
]
