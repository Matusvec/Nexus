using System;
using System.Collections.Generic;

namespace NexusXR.API.Models
{
    /// <summary>
    /// Represents a document in the Nexus knowledge base.
    /// Maps to the /documents response from the Nexus API.
    /// </summary>
    [Serializable]
    public class NexusDocument
    {
        public string id;
        public string filename;
        public string uploadedAt;
        public long fileSize;
        public int chunkCount;
        public string status; // "ready", "processing", "error"
        public string summary;
        public string groupId;
    }

    /// <summary>
    /// Response from the /upload endpoint.
    /// </summary>
    [Serializable]
    public class UploadResponse
    {
        public string id;
        public string filename;
        public string uploadedAt;
        public long fileSize;
        public int chunkCount;
        public string status;
        public string summary;
    }

    /// <summary>
    /// Response from DELETE /documents/{id}.
    /// </summary>
    [Serializable]
    public class DeleteResponse
    {
        public bool success;
        public string message;
    }

    /// <summary>
    /// Database-level statistics from the /stats endpoint.
    /// </summary>
    [Serializable]
    public class DatabaseStats
    {
        public int totalChunks;
        public int totalDocuments;
        public List<string> documents;
    }
}
