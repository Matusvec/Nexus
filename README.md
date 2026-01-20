# Nexus
Ultimate AI research helper

I will use chroma as my database
im going to try to explain my whole idea now, since that would proabbly be best for you now to help me project architecture, I want to build NEXUS: the ultimate AI research helper. when i was trying to do rag before, it ran into problems whenever i tried doing summary questions, because it wouldnt be able to do them since it used simple vector similarity search algorithms, VERY primitive, and then i found RAPTOR, and it looked amazing and offered a lot of versatility. So for the backend, I want to use a raptor hierarchy and create a tree. The big probem with raptor is the complexity with adding and removing document screening up the tree. 
For the broader project i want to be able to:
Have a beautiful graphic user interface, where they can drag and drop documents and order it however they want to on a blank canvas, they will be able to make groups of documents, query those groups using raptor, and then make subgroups in those groups, and go on making subgroups, and the chose to query outside of those groups, or link groups like amind map to query across connection lines, all with lighting fast, accurate, and quoted responses.
Chatgpt takes FOREVER to process documents and also has a hard limit on documents, this will support 1000s of documents in a robust database, and it will be lightning fast retrieval of data, plus there will be so much more, the AI will help you do ersearch along the way and each group will have saved your requests and will know about the connections and suggest ones, it will be able to organize these documents hwoever you would ever want, and it could be like an army of AI assistants, like with specialists in multple fields, like a mechanics AI, a physics AI, a coder Ai, a whatever ai, just assign groups, and each one will have a manager AI that will be an expert in the field, and they will be able to work together and create something incredible










[User Interface Layer]
  - Next.js App (App Router)
    - Infinite Canvas (React Flow) → groups, docs, connections
    - Chat/Query UI (shadcn + streaming)
    - AR/VR Teaser Mode (React Three Fiber + @react-three/xr)
[API / Business Logic Layer] (Next.js API Routes + Server Actions)
  - /api/upload → parse, chunk, embed, build RAPTOR tree per group
  - /api/query → retrieve (layer-aware, group-filtered), generate with Gemini
  - /api/groups → CRUD groups/connections (metadata updates)
[Data Layer]
  - ChromaDB (local persistent, collections per group/layer)
    - Chunks stored with metadata: { source_file, group_id, layer, parent_node_id (for future incremental) }
  - File System (for raw docs if needed, but prefer in-memory)
[AI Layer]
  - Gemini SDK (@ai-sdk/google)
    - Embeddings: embedding-001
    - Summarization & Generation: gemini-1.5-flash
    - Future: Tool calling for agent orchestration
[Runtime / Deployment]
  - Docker Compose: Next.js container + Chroma container
  - Local dev: npm run dev + Chroma in-memory or persisted




2. Key Components & Data Flow (Detailed)
A. Ingestion Pipeline (When User Uploads Docs)
Frontend: Drag-drop to canvas → assign to group (or default "Unsorted").
API: /api/upload
Parse: pdf.js or PyMuPDF wrapper (Node) for PDF; mammoth for DOCX.
Semantic chunking: Split by sentences/paragraphs, merge similar (cosine sim threshold ~0.7).
Embed chunks (Gemini embedding-001).
Build RAPTOR tree for that group:
Layer 0: Raw chunks + embeddings.
Layer 1+: Cluster (UMAP → HDBSCAN), summarize clusters with Gemini, embed summaries.
Limit: 2–3 layers, max 500 chunks per group for MVP speed.
Store in Chroma: One collection per group (e.g., raptor_group_physics), docs have metadata {group_id, layer, source_file, chunk_text}.
Mark group as "built" (metadata flag).
B. Canvas & Organization
React Flow nodes:
Doc nodes: Simple card (title, snippet).
Group nodes: Container (collapsible, color-coded).
Edges: User-drawn connections (store as metadata array on group docs: connected_group_ids).
On drag to new group: Update chunk metadata group_id → mark old/new groups dirty.
Save layout state (positions) to localStorage or simple JSON file in project dir.
C. Query Pipeline
User selects group(s)/connections → types query.
API: /api/query
Resolve target groups: Selected + connected (simple graph traversal on metadata).
For each group: If dirty → rebuild tree (lazy).
Retrieve: Query embeddings across layers (start high-level → drill down if needed).
Use Chroma query with filter {group_id: {$in: targets}, layer: {$gte: 0}}.
Augment prompt: Retrieved chunks + query + citations.
Generate: Gemini stream response with inline [citations].
Optional agent touch: If query spans groups → Gemini suggests "Connect physics to mechanics?" (tool call later).
D. AR/VR Teaser (Add in ~2–3 hrs if time)
Separate route /ar or button toggle.
Render same groups as 3D orbs (Sphere + shaderMaterial for glow/glass effect).
Positions: Map from React Flow coords (scale/offset).
Use Quest passthrough for mixed reality.
Test: Serve locally → open in Quest browser → enter XR session.
E. Handling Add/Remove (Lazy Group Rebuild)
On add doc: Chunk/embed → add to group's collection → mark group dirty.
On remove: Delete chunks matching source_file → mark dirty.
On query: If dirty → rebuild tree (re-cluster/summarize only that group's chunks).
Future: Add parent pointers for partial rebuilds.
F. Security/Local-First
Docker Compose: Next.js + Chroma (persist volume ./chroma_data).
Optional: Encrypt chunks (crypto-js AES) with user passphrase.
No external calls except Gemini (user key).

