using System.Collections.Generic;
using NexusXR.API.Models;

namespace NexusXR.API
{
    /// <summary>
    /// Provides realistic mock data for all API endpoints so the XR app
    /// can run without a live backend. All data mirrors the Nexus API contracts.
    /// </summary>
    public class MockDataProvider
    {
        private List<NexusDocument> _documents;

        public MockDataProvider()
        {
            InitializeDocuments();
        }

        // ── Documents ──────────────────────────────────────────────

        private void InitializeDocuments()
        {
            _documents = new List<NexusDocument>
            {
                new NexusDocument
                {
                    id = "doc_raptor_paper",
                    filename = "RAPTOR_Retrieval.pdf",
                    uploadedAt = "2026-01-15T09:00:00Z",
                    fileSize = 2456000,
                    chunkCount = 47,
                    status = "ready",
                    summary = "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. A hierarchical approach to RAG that builds multi-layer summaries.",
                    groupId = "group_research"
                },
                new NexusDocument
                {
                    id = "doc_motor_spec",
                    filename = "BrushlessMotor_Spec_v2.pdf",
                    uploadedAt = "2026-01-20T14:30:00Z",
                    fileSize = 890000,
                    chunkCount = 23,
                    status = "ready",
                    summary = "Technical specifications for a 48V brushless DC motor including torque curves, thermal limits, and mounting dimensions.",
                    groupId = "group_engineering"
                },
                new NexusDocument
                {
                    id = "doc_physics_notes",
                    filename = "QuantumTunneling_Lecture.pdf",
                    uploadedAt = "2026-01-22T11:15:00Z",
                    fileSize = 1200000,
                    chunkCount = 35,
                    status = "ready",
                    summary = "Lecture notes on quantum tunneling phenomena, including barrier penetration probability calculations and applications in semiconductor devices.",
                    groupId = "group_physics"
                },
                new NexusDocument
                {
                    id = "doc_api_design",
                    filename = "REST_API_BestPractices.md",
                    uploadedAt = "2026-02-01T16:00:00Z",
                    fileSize = 45000,
                    chunkCount = 12,
                    status = "ready",
                    summary = "Best practices for designing RESTful APIs including versioning, pagination, error handling, and authentication patterns.",
                    groupId = "group_engineering"
                },
                new NexusDocument
                {
                    id = "doc_processing",
                    filename = "NewUpload_InProgress.pdf",
                    uploadedAt = "2026-02-10T23:00:00Z",
                    fileSize = 3100000,
                    chunkCount = 0,
                    status = "processing",
                    summary = "",
                    groupId = ""
                }
            };
        }

        public List<NexusDocument> GetDocuments()
        {
            return new List<NexusDocument>(_documents);
        }

        public void RemoveDocument(string documentId)
        {
            _documents.RemoveAll(d => d.id == documentId);
        }

        // ── Query ──────────────────────────────────────────────────

        public QueryResponse GetQueryResponse(string question)
        {
            return new QueryResponse
            {
                answer = $"Based on the documents in your knowledge base, here is what I found regarding \"{question}\":\n\n" +
                         "RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a hierarchical tree of summaries from document chunks. " +
                         "Layer 0 contains the raw text chunks, while higher layers contain progressively more abstract summaries. " +
                         "This allows the system to answer both detail-oriented questions (using lower layers) and broad summary questions (using higher layers).\n\n" +
                         "The brushless motor specification indicates a peak torque of 2.4 Nm at 48V with a thermal limit of 85°C continuous operation.",
                sources = new List<RetrievalSource>
                {
                    new RetrievalSource
                    {
                        documentId = "doc_raptor_paper",
                        documentName = "RAPTOR_Retrieval.pdf",
                        chunkId = "chunk_raptor_014",
                        content = "RAPTOR constructs a tree by recursively clustering and summarizing text chunks. At the lowest level, the tree contains the original text segments. Each subsequent level groups related segments and generates an abstractive summary.",
                        layer = 0,
                        relevanceScore = 0.95f
                    },
                    new RetrievalSource
                    {
                        documentId = "doc_raptor_paper",
                        documentName = "RAPTOR_Retrieval.pdf",
                        chunkId = "chunk_raptor_032",
                        content = "The collapsed tree retrieval strategy traverses all layers simultaneously, selecting the most relevant nodes regardless of their position in the hierarchy. This outperforms layer-by-layer traversal on complex queries.",
                        layer = 1,
                        relevanceScore = 0.89f
                    },
                    new RetrievalSource
                    {
                        documentId = "doc_motor_spec",
                        documentName = "BrushlessMotor_Spec_v2.pdf",
                        chunkId = "chunk_motor_007",
                        content = "Peak torque: 2.4 Nm at 48V. Continuous torque: 1.8 Nm. Thermal protection activates at 105°C. Recommended continuous operating temperature: below 85°C for optimal bearing life.",
                        layer = 0,
                        relevanceScore = 0.72f
                    }
                },
                queryType = "complex",
                tokensUsed = 1842
            };
        }

        // ── Agent Run ──────────────────────────────────────────────

        public List<AgentEvent> GetAgentRunEvents(string task)
        {
            return new List<AgentEvent>
            {
                new AgentEvent
                {
                    type = "thinking",
                    content = $"Analyzing task: \"{task}\". I need to search the knowledge base and synthesize information from multiple documents.",
                    timestamp = 0f
                },
                new AgentEvent
                {
                    type = "tool_call",
                    toolName = "knowledge_search",
                    toolInput = $"{{\"query\": \"{task}\", \"top_k\": 5}}",
                    content = "Searching knowledge base...",
                    timestamp = 1.2f
                },
                new AgentEvent
                {
                    type = "tool_result",
                    toolName = "knowledge_search",
                    toolOutput = "Found 5 relevant chunks across 2 documents (RAPTOR_Retrieval.pdf, BrushlessMotor_Spec_v2.pdf)",
                    content = "Search complete. Found relevant information.",
                    timestamp = 2.8f
                },
                new AgentEvent
                {
                    type = "thinking",
                    content = "The search results cover the topic well. Let me also check if there are related documents that might provide additional context.",
                    timestamp = 3.5f
                },
                new AgentEvent
                {
                    type = "tool_call",
                    toolName = "list_documents",
                    toolInput = "{\"status\": \"ready\"}",
                    content = "Checking available documents...",
                    timestamp = 4.0f
                },
                new AgentEvent
                {
                    type = "tool_result",
                    toolName = "list_documents",
                    toolOutput = "4 documents available: RAPTOR_Retrieval.pdf, BrushlessMotor_Spec_v2.pdf, QuantumTunneling_Lecture.pdf, REST_API_BestPractices.md",
                    content = "Document inventory retrieved.",
                    timestamp = 4.8f
                },
                new AgentEvent
                {
                    type = "token",
                    content = "Based on my analysis of your knowledge base, here is a comprehensive answer:\n\n",
                    timestamp = 5.5f
                },
                new AgentEvent
                {
                    type = "token",
                    content = "The RAPTOR system uses a tree-structured approach to organize and retrieve information. ",
                    timestamp = 6.0f
                },
                new AgentEvent
                {
                    type = "token",
                    content = "It processes documents into chunks at Layer 0, then builds summary layers above. ",
                    timestamp = 6.5f
                },
                new AgentEvent
                {
                    type = "token",
                    content = "This allows both detailed fact retrieval and high-level summary queries from the same index.",
                    timestamp = 7.0f
                },
                new AgentEvent
                {
                    type = "done",
                    content = "Task completed successfully.",
                    timestamp = 7.5f
                }
            };
        }

        // ── Stats ──────────────────────────────────────────────────

        public DatabaseStats GetStats()
        {
            return new DatabaseStats
            {
                totalChunks = 117,
                totalDocuments = 4,
                documents = new List<string>
                {
                    "doc_raptor_paper",
                    "doc_motor_spec",
                    "doc_physics_notes",
                    "doc_api_design"
                }
            };
        }
    }
}
