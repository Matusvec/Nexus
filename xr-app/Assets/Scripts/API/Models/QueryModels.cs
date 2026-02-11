using System;
using System.Collections.Generic;

namespace NexusXR.API.Models
{
    /// <summary>
    /// A single retrieval result with citation information.
    /// Maps to the /query response source objects from the Nexus API.
    /// </summary>
    [Serializable]
    public class RetrievalSource
    {
        public string documentId;
        public string documentName;
        public string chunkId;
        public string content;
        public int layer;
        public float relevanceScore;
    }

    /// <summary>
    /// Full query response from the Nexus /query endpoint.
    /// </summary>
    [Serializable]
    public class QueryResponse
    {
        public string answer;
        public List<RetrievalSource> sources;
        public string queryType;
        public int tokensUsed;
    }

    /// <summary>
    /// Request body for the /query endpoint.
    /// </summary>
    [Serializable]
    public class QueryRequest
    {
        public string question;
        public string document_id;
        public string group_id;
        public int top_k = 10;
        public string persona_id;
    }
}
