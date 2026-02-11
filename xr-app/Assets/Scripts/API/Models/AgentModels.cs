using System;
using System.Collections.Generic;

namespace NexusXR.API.Models
{
    /// <summary>
    /// Represents a single event in an agent run stream.
    /// Used for both real SSE events and mock simulation.
    /// </summary>
    [Serializable]
    public class AgentEvent
    {
        public string type; // "thinking", "tool_call", "tool_result", "token", "done", "error"
        public string content;
        public string toolName;
        public string toolInput;
        public string toolOutput;
        public float timestamp;
    }

    /// <summary>
    /// Request body for starting an agent/orchestrator run.
    /// </summary>
    [Serializable]
    public class AgentRunRequest
    {
        public string task;
        public string persona_id;
        public string document_id;
        public string group_id;
    }

    /// <summary>
    /// Summary of a completed agent run.
    /// </summary>
    [Serializable]
    public class AgentRunSummary
    {
        public string runId;
        public string task;
        public string status; // "running", "completed", "failed"
        public string result;
        public List<AgentEvent> events;
        public float durationSeconds;
        public string startedAt;
    }
}
