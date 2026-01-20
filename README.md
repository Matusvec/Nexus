# Nexus
**Your Personal Team of AI Research Specialists**

> Transform document chaos into an intelligent, collaborative AI workspace. Organize 1000s of documents, query with human-like understanding, and delegate work to specialist AI personas that know when to ask for your help.

---

## The Vision

ChatGPT can't handle complex research. It has file limits, slow processing, and fails at summary questions. **Nexus is different.**

- **Infinite canvas** where you drag, organize, and connect document groups like a mind map
- **RAPTOR retrieval** that understands both details and high-level summaries
- **Specialist AI personas** (Max the mechanic, Dr. Elena the physicist, Byte the coder) that collaborate across domains
- **Human-in-the-loop** delegation where agents request physical tasks with step-by-step guidance
- **Lightning fast**, **fully local**, supports **1000s of documents**
- **AR/VR mode** to walk among your knowledge groups in 3D space

---

##  Core Features

### 1. Intelligent Canvas Organization
- Drag-drop documents onto infinite 2D canvas
- Create **groups** and **subgroups** (nested hierarchies)
- Draw **connection lines** between groups (mind-map style)
- Query individual groups, connected components, or entire workspace
- Visual layout persists (your spatial memory = knowledge map)

### 2. RAPTOR Hierarchical Retrieval
- **Layer 0**: Raw document chunks (detailed facts)
- **Layer 1+**: AI-generated summaries of clustered chunks (high-level concepts)
- Automatically selects layer based on query type:
  - "What did the paper say about X?" → Layer 0 (details)
  - "Summarize all documents on Y" → Layer 2 (summaries)
- Handles 1000s of documents with lazy tree rebuilding

### 3. Specialist AI Personas
Each group can be assigned specialist agents with **custom personalities**:

| Persona | Role | Personality | Example Response |
|---------|------|-------------|------------------|
| **Max** | Mechanical Engineer | Gruff, practical, safety-focused | *"Alright kid, that bearing's gonna fail. You using 6061 or 7075?"* |
| **Dr. Elena** | Physicist | Precise, encouraging, explains deeply | *"Interesting! The tunneling probability depends on barrier width..."* |
| **Byte** | Software Engineer | Fast-talking, meme references | *"Bruh, your loop is O(n²). Let me refactor..."* |
| **Stacy** | Electrical Engineer | Methodical, diagram-obsessed | *"Let's trace the signal path. Ground loop issue at C3..."* |

- Talk naturally: *"Hey Max, will this motor mount handle 5G acceleration?"*
- Agents collaborate: *"Max here - pulling in Elena to check the electromagnetics"*
- Fully customizable: names, tones, quirks, humor levels

### 4. Human-in-the-Loop Delegation
Agents **know their limits** and explicitly request human help:

```
Max: "I need you to measure the shaft diameter. Here's how:
      1. Grab your calipers (±0.01mm accuracy minimum)
      2. Measure at 3 points along the length
      3. Take a photo showing the reading
        Safety: Deburr sharp edges first
      Upload when done and I'll continue the analysis."
```

- **Structured task requests** (3D printing, soldering, code testing, measurements)
- **Step-by-step instructions** with safety warnings
- **Upload interface** for human responses (files, photos, data)
- **Workflow continuity** - agent resumes after human completes task

### 5. Agentic Tool Use
Agents use tools autonomously (not dumb classification):
- `query_group(group_id, query, layer)` - RAPTOR retrieval
- `get_connected_groups(group_id)` - Navigate mind map
- `search_all_groups(query)` - Find relevant specialists
- `request_human_task(task_spec)` - Delegate physical work
- `suggest_connection(group_a, group_b, reason)` - Propose links
- Future: `web_search()`, `run_calculation()`, `generate_visualization()`

### 7. Interactive Learning Mode

**Socratic Method + Visual Explanations**

Agents don't just answer - they teach through dialogue and demonstration:

```
You: "Why not steel instead of aluminum?"

Max: *Walks to whiteboard, spawns material properties chart in AR*
     "Good question! Let me show you..."
     
     *Highlights strength-to-weight ratios*
     "Steel IS stronger, but we're mounting on a drone. Watch..."
     
     *Spawns two AR cubes labeled "aluminum" and "steel"*
     *Shows mass: 35g vs 95g*
     
     "That extra 60 grams eats 15% of flight time.
     And at these stress levels..." *shows force diagram*
     "...7075 has a 3x safety margin. Steel is overkill."

You: "What about cost?"

Max: "Fair point!" *Pulls up pricing chart*
     "7075 is $8/part vs steel at $3, but machining time..."
     *Explains trade-offs with diagrams*
     "Total cost is basically a wash. Still prefer aluminum for weight."

You: "Okay, I'm convinced."

Max: "Good reasoning! You questioned assumptions, weighed trade-offs.
     That's engineering thinking."
```

**Benefits:**
- Learn by doing and debating, not memorizing
- Visual explanations (charts, diagrams, 3D models appear in space)
- Context-aware teaching (knows your project, tailors explanation)
- Agents remember what you've learned (build on previous lessons)

### 8. Educational Applications (Future Vision)

**Nexus Learn - Making Education Spatial and Fun**

The same architecture enables transformative educational experiences for K-12 students:

**Adaptive Persona Examples:**

**Professor Pete (Math AI):**
```
Kid: "Ugh, I don't wanna do fractions!"

Pete: *Appears as friendly cartoon character*
      "Yo, I heard you like pizza..."
      
      *Spawns holographic pizza on kitchen table*
      "Your friend wants half. Show me how to cut it."
      
Kid: *Uses AR knife to cut pizza*

Pete: "Nice! That's 1/2. But what if TWO friends come over?"
      *Pizza reforms*
      "Now cut it for three people."
      
Kid: *Cuts into thirds*

Pete: "BOOM! That's 1/3. You just learned fractions!
      Now catch three floating pizza slices..." *Gamifies learning*
```

**Coach Rex (Physics through Sports):**
- Teaches projectile motion using AR basketball trajectories
- "Wanna dunk? You need parabolic motion!"
- Spawns AR basketball court in living room

**Captain Astro (Space Explorer):**
- Teaches astronomy by spawning solar system in bedroom
- Kid "flies" to planets, experiences scale
- "That's 93 million miles to the Sun - takes light 8 minutes!"

**Real-World Learning:**
```
Nova (Music AI): "See those Lego bricks on your desk?"
                 *Highlights 3 piles of 4 bricks with AR glow*
                 
                 "Count 'em. 4, 4, 4... that's 4+4+4 = 12!
                 That's also 3 × 4 = 12!
                 
                 Now build me a tower of 5×3.
                 Winner designs my next outfit!"
                 
Kid: *Actually wants to do math*
```

**Parent Dashboard:**
- Time spent with each tutor
- Concepts mastered vs struggling
- Engagement scores (compared to traditional methods)
- Recommended next topics

**Note:** Educational mode is a future expansion. Current focus is professional research tool (Nexus Pro).

### 6. AR/VR Spatial Coworker Mode

**Transform your workspace into an AI-powered collaborative environment.**

Instead of viewing floating orbs, you work alongside spatially-embodied AI specialists in your physical workspace:

**Spatial Presence:**
- Agents appear at designated stations in your room (workbench, desk, whiteboard, electronics bench)
- Each agent "stands" at their specialty area
- Walk between agents like visiting coworkers on a factory floor
- Positional audio - agents speak from their location, volume based on distance

**Interactive Workflow:**
```
You're at your desk coding...

Max (appears at CAD workbench, 10 feet away):
  "Hey, come check out this motor mount design."
  
[You walk over, Max highlights workbench with AR glow]
[3D holographic model appears above bench]

Max: "See this stress point?" *Points with AR raycast*
     "I need you to 3D print this revised version."
     *STL downloads to your printer*
     "Should take 2 hours. I'll keep working on the frame."

[You return to desk]

Elena (appears at electronics bench, soft notification chime):
  "When you get a moment, could you come to the soldering station?
   Need you to solder the ESC."

[Walk to electronics bench]
[Quest highlights soldering iron with green outline]
[PCB schematic overlays on real board with AR dots showing points]

Elena: "Joint #3 needs more flux." *Arrow points to exact spot*
       "Perfect! Now bridge these two pads..."
```

**Real-World Object Integration:**
- Quest's scene understanding detects surfaces (desk, workbench, whiteboard)
- Agents reference and highlight real objects ("grab your calipers", "the soldering iron")
- AR overlays on physical items (measurement guides, assembly instructions, safety warnings)
- Upload interface for human task completion (photos, measurements, files)

**Agent Behaviors:**
- **Working state:** Animated typing, tool interactions, reading documents
- **Needs attention:** Gentle chime, wave gesture, waiting pose
- **Urgent:** Red glow, insistent audio cue
- **Idle:** Leaning, casual animations, reviewing overnight work

**Session Continuity:**
- Agents remember positions and tasks between sessions
- "Morning! Ready to finish that motor assembly?" on startup
- Save/resume project state with spatial context intact

This is **Iron Man's JARVIS workshop** - AI team members working alongside you in physical space.

---

##  Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│  Next.js App (App Router) + React + TypeScript              │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │ Canvas Mode  │  │  Chat UI   │  │  AR/VR Mode      │   │
│  │ (React Flow) │  │ (shadcn)   │  │ (React Three XR) │   │
│  └──────────────┘  └────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND API LAYER                          │
│  FastAPI (Python) - preferred for ML/clustering             │
│  OR Next.js API Routes (if staying Node-only)               │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐ │
│  │ /upload   │  │  /query  │  │  /agents  │  │  /tasks  │ │
│  │ (ingest)  │  │ (agentic)│  │  (CRUD)   │  │ (human)  │ │
│  └───────────┘  └──────────┘  └───────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                   AGENT ORCHESTRATION                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Meta-Orchestrator (routes to specialists)          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐          │
│  │ Max    │  │ Elena  │  │ Byte   │  │ Stacy  │  ...     │
│  │ (Mech) │  │(Physics)│  │ (Code) │  │ (EE)   │          │
│  └────────┘  └────────┘  └────────┘  └────────┘          │
│      ↕            ↕            ↕            ↕              │
│  [Tool Layer: query_group, request_human_task, etc.]      │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ChromaDB (local persistent)                         │  │
│  │  - Single collection with metadata filtering         │  │
│  │  - Chunks: {group_id, layer, parent_id, source_file} │  │
│  │  - Embeddings: Gemini embedding-001                  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Agent State Store (SQLite or JSON)                  │  │
│  │  - Persona configs, conversation history             │  │
│  │  - Human task queue, group connections               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                       AI LAYER                               │
│  Gemini SDK (@google/generative-ai or @ai-sdk/google)      │
│  - Embeddings: text-embedding-004                           │
│  - Generation: gemini-2.0-flash (tool calling support)     │
│  - Summarization: gemini-1.5-flash (clustering summaries)  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow: Document Upload → Query

**Upload Pipeline:**
```
1. User drags PDF to canvas → assigns to group "Physics"
2. Backend parses (pdf.js), chunks semantically
3. Embed chunks with Gemini
4. Build RAPTOR tree:
   - Layer 0: Store chunks
   - Layer 1: Cluster (UMAP→HDBSCAN), summarize clusters
   - Layer 2: Cluster summaries, create meta-summaries
5. Store in ChromaDB with metadata: {group_id: "physics_001", layer: 0/1/2}
```

**Query Pipeline:**
```
1. User: "Hey Max, summarize all motor datasheets"
2. Orchestrator identifies Max (mechanics specialist)
3. Max uses tool: query_group(group_id="mechanics", layer=2)
4. RAPTOR retrieves high-level summaries
5. Max generates answer with personality
6. Streams response with citations
```

**Human-in-Loop Pipeline:**
```
1. User: "Max, can this mount handle vibration?"
2. Max analyzes docs, realizes needs physical test data
3. Max uses tool: request_human_task({
     type: "measurement",
     instructions: "Mount accelerometer at point A...",
     safety: ["Secure fixture before powering on"],
     expected_output: "CSV file with acceleration data"
   })
4. UI shows task card with upload interface
5. User completes task, uploads data
6. Max ingests data, continues analysis
```

---

## Development Strategy

### Validation-First Philosophy

**Core Principle:** Build the minimum to prove each hypothesis. Don't assume users want features. Make them demand it.

**Kill everything except the core value proposition, then add back only what's proven necessary.**

---

## Phase 0: Kill the UI. Kill AR. Kill Personas.

**Build only this:**
- CLI or minimal web UI (bare bones)
- Upload documents
- RAPTOR-style hierarchical retrieval
- One neutral agent (no personality)

**One killer demo:**
> "Ask a question that normal RAG gets wrong — Nexus gets right."

**Success Metric:**
- Can you dominate side-by-side comparisons with standard RAG?
- If NO → the rest is irrelevant, fix this first
- If YES → proceed to Phase 1

**Timeline:** 2-3 months

**Why this matters:** RAPTOR is your technical moat. Prove it works before adding anything else.

---

## Phase 1: Add One Agent + Human-in-Loop

**Only after RAPTOR clearly outperforms alternatives:**

**Add:**
- One specialist agent (neutral, professional tone)
- Human task request system
- Tool calling (query_group, request_human_help)

**Goal:** Prove the agent stops instead of hallucinating

**Test:**
- Give agent questions it cannot answer from documents
- Does it say "I need you to..." or does it make things up?
- This is where your real differentiation appears

**Success Metric:**
- Agent reliably requests human help when needed
- Users say: "This doesn't hallucinate like ChatGPT"
- 10 people willing to pay $10 for early access

**Timeline:** 2 months

**Decision Point:** If agents don't add value over plain RAPTOR search, don't build personas/multi-agent.

---

## Phase 2: Canvas UI (Only If Users Ask)

**Do not assume spatial organization is good. Prove it.**

**Wait for users to say:**
- "I lose track of document groups"
- "I want to see relationships between topics"
- "Can I organize this visually?"

**Then and only then:**
- Build basic canvas (React Flow)
- Boxes for groups, lines for connections
- No fancy animations, no AR teases, just functional

**Success Metric:**
- Users naturally organize documents spatially
- They return to the canvas view repeatedly
- Retention improves after canvas is added

**Timeline:** 2-3 months

**If users don't ask:** Stay with list/folder view. Simpler is better.

---

## Phase 3: Personas (Lightweight)

**Only after users trust correctness.**

**Personality should feel like:**
- "Oh nice, this makes it easier to think."

**Not:**
- "This is distracting but cute."

**Implementation:**
- Start with tone variations (formal vs casual)
- Add names only if users request them
- "Max the mechanic" comes after people want specialist contexts

**Test:**
- A/B test: neutral vs personality
- Does personality improve or hurt retention?
- Do users engage more or complain?

**Success Metric:**
- Personality doesn't reduce trust
- Users give agents nicknames organically
- "Can I talk to the physics expert?" appears in feedback

**Timeline:** 1-2 months

**If personality reduces trust:** Stay neutral. Accuracy > entertainment.

---

## Phase 4: AR (Much Later, Maybe Never)

**AR is a distribution amplifier, not a foundation.**

**Build AR when:**
- You have 1000+ paying users
- Users are begging for spatial interaction
- You have funding to hire 3D/AR specialist
- 2D product is excellent and profitable

**AR serves two purposes:**
1. **Marketing/PR:** TechCrunch headlines, demo videos, VC interest
2. **Power users:** Small subset who think spatially

**Reality Check:**
- <5% of users will have Quest headsets
- 2D product must work standalone
- AR development is 6+ months
- Don't build AR to "differentiate" - build it when users demand it

**If you ever reach this phase, you'll know — users will be requesting it constantly.**

---

## Target Market (Refined)

**Your real customers are NOT:**
- Students (low budget, low commitment)
- Casual ChatGPT users (don't need local)
- Productivity tourists (churn after 1 month)

**Your real customers ARE:**
- Engineers with 100+ datasheets
- PhD students drowning in papers
- Startup founders doing market research
- Research-heavy professionals (legal, medical, policy)
- Serious makers with deep project documentation

**This is a small but high-value market:**
- Willing to pay $30-50/month
- Need privacy (can't use ChatGPT for proprietary data)
- Value accuracy over speed
- Use daily, not occasionally

**Market Size:**
- Addressable: ~500K potential users globally
- Realistic: 5-10K paying users within 2 years
- Revenue potential: $1.8-5.4M ARR (at $30/mo)

**This is enough to build a profitable, sustainable business.**
Practical Considerations

### Scope Realities

**What You're Building:**
- Advanced RAG system (RAPTOR hierarchical retrieval)
- Multi-agent orchestration with personalities
- Human-in-loop task management
- Infinite canvas interface
- Spatial computing AR/VR mode
- Real-world object interaction
- Educational applications

**Honest Assessment:** This is 3-4 different products. The key is building incrementally and validating at each stage.

### Key Challenges

**1. RAPTOR Update Problem**
- OKey Innovations

### Novel Contributions

1. **Spatial Coworker Paradigm**
   - AI agents embodied in physical workspace (not floating screens)
   - Walk between specialists like a factory floor manager
   - Real-world object interaction and highlighting
   - Potentially patent-worthy UX design

2. **Human-in-Loop Delegation**
   - Agents explicitly know their limits
   - Structured physical task requests with safety guidance
   - Workflow continuity across digital/physical boundaries
   - Novel interaction pattern for AI collaboration

3. **Personality-Driven Multi-Agent**
   - Research team with names and quirks, not faceless chatbots
   - Reduces cognitive load through memorable personas
   - Natural language interface ("Hey Max, can you...")
   - Makes complex system approachable

4. **Hierarchical RAG with Spatial Organization**
   - RAPTOR retrieval + visual canvas mind-mapping
   - Group-level query filtering with connection traversal
   - Layer-aware retrieval (details vs summaries)
   - Lazy tree rebuilding for practical document updates

5. **Interactive Learning Mode**
   - Socratic method with spatial visual aids
   - Agents teach through debate and demonstration
   - Context-aware explanations tied to user's projects
   - Bridges professional tool and educational applications

6. **Local-First Privacy**
   - All processing on user hardware
   - No data leaves device without explicit consent
   - Differentiator vs cloud-only tools (ChatGPT, Perplexity)
   - Critical for enterprise/research sensitive data

---

## Design Principles

**User as CEO**
- Agents are assistants that request permission and guidance
- No autonomous actions without human approval
- User maintains control and understanding of system

**Personality = Interface**
- Names and personas reduce cognitive complexity
- Memorable characters easier than group/collection IDs
- Quirks create emotional connection and engagement

**Spatial Memory**
- Visual canvas leverages human spatial reasoning
- Physical positioning in AR mode matches mental models
- Location = context (walk to workbench = CAD work)

**Explicit Limitations**
- Agents clearly state what they cannot do
- Request human help with structured guidance
- No pretending capabilities (unlike general chatbots)

**Local by Default**
- Privacy through edge computing
- Speed through local processing
- Trust through transparency (you control the data)

**Progressive Enhancement**
- 2D canvas works without AR
- Single agent works without multi-agent
- CLI works without UI
- Each layer adds value without requiring previous layers

---

## Future Expansion

### Phase 5+ (After Product-Market Fit)

**Mobile AR App**
- iPhone/iPad with ARKit
- Scan room, place agent orbs
- Voice-first interaction
- Lightweight compared to Quest

**Voice-First Interaction**
- AirPods + spatial audio
- Hands-free querying while working
- "Hey Max, what's the torque spec?" while holding wrench
- Ambient computing paradigm

**Collaborative Mode**
- Shared workspaces between team members
- Multi-user document collections
- Team chat with AI specialists
- Enterprise collaboration features

**Plugin Ecosystem**
- Custom tools for agents (Zapier-style)
- Domain-specific integrations (CAD software, lab equipment)
- Community-contributed agent personas
- Marketplace for specialized agents

**Enterprise Features**
- On-premises deployment
- SSO/SAML authentication
- Audit logging
- Role-based access control
- Compliance certifications (SOC2, HIPAA)

**Nexus Learn Product Line**
- K-12 curriculum-aligned content
- Parent/teacher dashboards
- School district deployment
- Gamification and achievement systems
- Accessibility features (dyslexia support, translations)

---

## Contributing

**Current Status:** Early development, not yet open for contributions.

**Future Plans:**
- Open source core after Milestone 3
- Contribution guidelines for agent personas
- Plugin API for custom tools
- Community persona library

---

## FAQ

**Q: When will this be available?**  
A: Phase 0 (CLI RAPTOR demo) in ~2-3 months. Each phase after depends on validation results. AR is Phase 4+ (maybe never if users don't request it).

**Q: How much will it cost?**  
A: Early adopters: $10-15/month during beta. Later: $30-50/month for professionals. Free tier possible for open source core.

**Q: What hardware do I need?**  
A: Desktop/laptop. That's it. AR mode is Phase 4+ and completely optional.

**Q: Will my data be private?**  
A: Yes. Everything runs locally. Only API calls are to Gemini for embeddings/generation (you provide your own API key).

**Q: Can I use my own LLM instead of Gemini?**  
A: Future feature. Initial version uses Gemini, but architecture supports swapping models.

**Q: Does this replace ChatGPT?**  
A: Different use case. ChatGPT: general questions. Nexus: deep research on YOUR documents with accurate citations and no hallucinations.

**Q: Why not just use ChatGPT with file uploads?**  
A: ChatGPT has 50-file limit, forgets context, hallucinates, and can't handle summary questions well. Nexus uses RAPTOR for hierarchical understanding and admits when it doesn't know.

**Q: What about fancy personas and AR?**  
A: Only if validation proves users want them. Phase 0 is pure RAPTOR. Everything else is conditional on user demand.

---

## License

MIT License (intended - to be added upon first release)

---

## Project Status

**Current Phase:** Pre-Development (Architecture & Planning)  
**Next Phase:** Phase 0 - Prove RAPTOR Works (2-3 months)  
**Current Focus:** Kill all unnecessary features, build only the core value proposition

**Looking For:** 
- Early beta testers (engineers/researchers with 100+ documents in their workflow)
- People who can provide honest feedback: "Does this beat standard RAG?"
- Users willing to do side-by-side comparisons

**Not Looking For (Yet):**
- UI/UX designers (no UI in Phase 0)
- AR/VR developers (Phase 4+, maybe never)
- Growth hackers (need product-market fit first)

---

**Contact:** [To be added]  
**GitHub:** [Repository to be created]  
**Website:** [To be built]  
**Twitter:** [Updates to be postedmonths
- Get real user feedback continuously
- Be willing to pivot based on data

**5. AR Technical Limitations (Quest 3 in 2026)**
- Passthrough: Excellent
- Hand tracking: Reliable
- Spatial anchors: Good with setup
- Scene understanding: Basic (detects planes, not specific objects)
- Object recognition: Limited without custom ML models
- Real-time arbitrary object highlighting: Not available

**Achievable AR for MVP:**
- Agents at fixed positions (user defines stations manually)
- Holographic content floating in space
- Voice interaction
- Gesture controls
- Generic location references ("look at your desk") vs specific object detection ("pick up the soldering iron")

Advanced object detection requires custom ML models (add 3-4 months development).

---

## Business Model & Economics

### Pricing Strategy

**Target Market: Professional researchers, engineers, makers**

**Options:**
1. **Lifetime License:** $299 (early adopter pricing)
   - Get cash flow immediately
   - Use for first 500 customers
   - Creates loyal early community

2. **Subscription:** $30/month or $299/year
   - Recurring revenue for sustainability
   - Switch after proven market fit
   
3. **Self-Hosted Free + Hosted Paid**
   - Open source core for trust/adoption
   - Charge for managed hosting/support
   - Developer-friendly business model

### Cost Structure

**API Costs (Gemini):**
- Embeddings: $0.001/1K tokens
- 1 PDF (10K tokens) = $0.01 to embed
- 1000 PDFs per user = $10 setup cost
- Query costs: ~$0.0005/query
- Heavy user (100 queries/day) = $1.50/month

**Margins at $30/month:**
- API costs: ~$2/month per user
- Gross margin: ~93%
- Excellent SaaS economics

**Hosting:**
- Local-first: User provides compute (zero hosting cost)
- Cloud option: $20-50/user/month (kills margins, avoid initially)

**Recommendation:** Local-first is correct business decision
- Users run on their hardware
- You provide software + updates
- Keeps costs low, margins high

### Revenue Projections (Realistic)

**Year 1 (MVP + Beta):**
- 100 paying beta users × $10/month = $12K annual revenue
- Covers API overages, not salary
- Focus: Validation and refinement

**Year 2 (Public Launch):**
- 1,000 users × $30/month = $360K ARR
- Support 1 developer salary + 1 hire
- Focus: Growth and stability

**Year 3 (Scaling):**
- 5,000 users × $30/month = $1.8M ARR
- Team of 4-5 people
- Profitable, sustainable
- Focus: Polish and expansion

### Funding Path

**Bootstrap (Recommended Initial Path):**
- Need ~$100K personal runway (living expenses for 12 months)
- Ship Milestone 3, prove retention
- Reach $15K MRR before raising

**Seed Round ($500K-750K):**
- Raise after Milestone 3 (working UI + paying users)
- Use to hire 2 developers
- Accelerate to AR prototype (Milestone 5)
- 18-month runway to profitability

**Series A ($3-5M):**
- After reaching $500K ARR with strong retention
- Hire full team (10-15 people)
- Build educational version (Nexus Learn)
- Expand to enterprise market

---

## Key Risks & Mitigation

### Risk 1: Build for 2 Years, No One Cares
**Probability:** High if building in isolation  
**Impact:** Catastrophic (wasted time/money)  
**Mitigation:** Ship every 3 months, get real user feedback, validate continuously

### Risk 2: Apple Vision Pro Adds Similar Features
**Probability:** Medium (they're focused on $3500 enterprise market)  
**Impact:** Medium (market perception)  
**Mitigation:** Target $299 prosumer market, local-first privacy as moat

### Risk 3: OpenAI Integrates This into ChatGPT
**Probability:** Medium (they have resources)  
**Impact:** High (market validation but fierce competition)  
**Mitigation:** They can't do local-first + privacy. That's your defensible moat. Speed to market matters.

### Risk 4: Developer Burnout
**Probability:** High (ambitious scope, solo developer)  
**Impact:** High (project stalls)  
**Mitigation:** Ship smaller scope. Get revenue sooner. Hire help earlier. Celebrate small wins.

### Risk 5: AR is Gimmick, Users Prefer 2D
**Probability:** Medium-High (current VR adoption rates)  
**Impact:** Medium (wasted AR dev time)  
**Mitigation:** Make 2D product excellent first. AR is enhancement, not dependency.

### Risk 6: Education Market Too Different
**Probability:** High (very different GTM, regulations, sales cycles)  
**Impact:** Medium (distraction from core)  
**Mitigation:** Focus exclusively on professional market for first 2 years. Defer education.

---

## Strategic Recommendations

### What to Build: Nexus Pro MVP

**In Scope:**
- RAPTOR retrieval (simplified: 2 layers, lazy rebuild, 100 doc groups)
- 3 specialist agents (Max, Elena, Byte)
- Human-in-loop task system
- Simple canvas UI (functional over fancy)
- Local-first, self-hosted
- Chat interface with streaming + citations

**Out of Scope for MVP:**
- AR/VR mode (build after revenue)
- Education market (future product line)
- Advanced object detection (tech not ready)
- Multi-user collaboration (unnecessary complexity)
- Mobile apps (desktop first)

**Timeline: 9 months to MVP, 6 months to first revenue**

### Go-to-Market Plan

**Launch Strategy:**
1. Build in public (blog posts, Twitter threads, GitHub)
2. Ship Milestone 1 → Post on HackerNews
3. Ship Milestone 2 → Demo video on YouTube
4. Ship Milestone 3 → Launch on Product Hunt

**Target Communities:**
- Hackaday (makers/engineers)
- r/AskEngineers, r/MachineLearning (Reddit)
- IndieHackers (bootstrapped founders)
- Hacker News "Show HN"

**Pricing:**
- First 500 users: $299 lifetime (early adopter)
- After validation: $30/month or $299/year subscription

### Success Metrics

**Month 3 (Milestone 1):**
- You use CLI daily
- 100+ GitHub stars or waitlist signups

**Month 6 (Milestone 2):**
- 5 people pay $10 for early access
- Daily active usage from beta users

**Month 9 (Milestone 3):**
- 50 beta users
- 20%+ weekly retention rate
- Qualitative feedback: "I can't work without this"

**Month 12:**
- 500 paying users
- $15K monthly recurring revenue
- Break-even on costs

**Month 18:**
- 1,500 users
- $45K MRR
- Begin AR development (Milestone 5)

**Month 24:**
- 3,000 users
- $90K MRR
- Team of 3-4 people
- AR prototype shipped

---

## Product Roadmap

### Two-Product Strategy (Long-Term Vision)

**Nexus Pro** (Phase 1-3)
- Target: Engineers, researchers, makers, professionals
- Use case: Complex projects with multi-domain knowledge
- Pricing: $30/month or $299 lifetime
- Market: Niche but high-value (100K potential users)

**Nexus Learn** (Phase 4+, Future)
- Target: K-12 students, homeschoolers, struggling learners
- Use case: Interactive spatial learning with AI tutors
- Pricing: $10/month per student, school licenses
- Market: Mass market (50M+ potential students)

**Shared Infrastructure:**
- Same agent system (different personas)
- Same spatial computing technology
- Same ChromaDB architecture
- Same RAPTOR retrieval

**Strategy:** Build Pro first (18 months), prove concept, generate revenue. Then expand to Learn market with proven technology.

---

## Getting Started

### Current Status
- [x] Basic ChromaDB wrapper (`database.py`)
- [x] Project vision and architecture defined
- [ ] RAPTOR tree implementation
- [ ] Agent system
- [ ] Canvas UI
- [ ] AR mode

### Immediate Next Steps (Week 1)

**Goal: Prove basic RAPTOR works**

1. Set up FastAPI backend structure
2. Implement semantic chunking (sentence splitting)
3. Integrate Gemini embeddings
4. Build 2-layer RAPTOR tree (no clustering yet, just hierarchy)
5. Create CLI query interface
6. Test with 5-10 documents

**Success Criteria:** You can query documents and get better answers than simple keyword search

### Sprint 1 (Month 1)

1. Add UMAP + HDBSCAN clustering
2. Implement cluster summarization with Gemini
3. Build full RAPTOR tree (Layer 0 chunks → Layer 1 summaries)
4. Query layer selection logic
5. Citation extraction
6. Blog post with before/after examples

**Success Criteria:** Demo shows clear improvement on summary questions vs basic RAGor, but 2D remains core product

**Total Timeline: 14 months to working AR prototype with validated core product**

---

## Development Phases (Original Optimistic Plan)

**Reality Check:** Initial phase estimates were too optimistic. Adjusted timeline above reflects realistic solo developer pace.

### Phase 1: Core RAPTOR + Single Agent (4-6 months realistic)
- FastAPI backend + ChromaDB + Gemini integration
- Document ingestion pipeline (PDF parsing, chunking, embedding)
- RAPTOR tree builder (2 layers, lazy rebuild)
- Single specialist agent (Max) with personality + tools
- Basic 2D canvas UI (React Flow, drag-drop groups)
- Chat interface with streaming responses
- Citation/quote extraction

### Phase 2: Multi-Agent + Human-in-Loop (3-4 months)
- Multi-specialist system (3-4 personas initially)
- Persona customization UI (names, tones, quirks)
- Human task queue + upload interface
- Inter-agent communication (message passing)
- Cross-group queries via connections
- Group-level memory (conversation history per agent)

### Phase 3: AR/VR Spatial Coworker Mode (6-8 months)
- React Three Fiber + @react-three/xr setup
- Spatial agent positioning at workstations
- Quest passthrough integration
- Voice query interface (Web Speech API)
- Spatial audio (positional sound)
- Basic scene understanding (surface detection)
- Real-world object highlighting (simplified - user defines objects)
- Agent animations and behaviors
- Sync 2D/3D canvas state

### Phase 4: Polish + Advanced Features (3-4 months)
- OCR + layout-aware parsing (images, tables)
- Web search tool for agents
- Visualization generation (charts, diagrams)
- Agent suggestion system (connection recommendations)
- Advanced RAPTOR (incremental updates via parent pointers)
- Encryption (AES for chunk storage)
- Docker deployment
- Multi-user support (optional)

**Total Realistic Timeline:**
- Solo developer: 20-27 months to production-ready AR version
- With 3-person team: 12-15 months
- To first paying users (Milestone 3): 8-9 months

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | Next.js 14 + TypeScript | App Router, Server Components, modern DX |
| **Canvas** | React Flow | Infinite canvas, node graphs, mature library |
| **UI** | shadcn/ui + Tailwind | Beautiful, accessible components |
| **AR/VR** | React Three Fiber + XR | Declarative 3D, Quest support |
| **Backend** | FastAPI (Python) | Better ML/clustering support than Node |
| **Database** | ChromaDB | Local-first vector DB, persistent |
| **State** | SQLite / JSON files | Agent configs, task queue, connections |
| **AI** | Gemini 2.0 Flash | Fast, tool calling, affordable embeddings |
| **Clustering** | UMAP + HDBSCAN | Standard for RAPTOR implementation |
| **Deployment** | Docker Compose | Isolated services, easy distribution |

**Alternative (Node-only stack):**
- Backend: Next.js API Routes + Server Actions
- Clustering: TensorFlow.js + custom HDBSCAN port (harder)

---

## 🎬 Getting Started

### Current Status
- [x] Basic ChromaDB wrapper (`database.py`)
- [ ] RAPTOR tree implementation
- [ ] Agent system
- [ ] Canvas UI
- [ ] AR mode

### Next Steps (Phase 1 Sprint 1)
1. **Set up project structure** (FastAPI + Next.js frontend)
2. **Implement RAPTOR tree builder** (clustering + summarization)
3. **Create first specialist agent** (Max with personality)
4. **Build document ingestion pipeline** (PDF → chunks → embeddings → tree)
5. **Basic canvas UI** (drag-drop, group creation)

---

## 🌟 Key Innovations

1. **Human-in-Loop Delegation** - Agents that know when to ask for help (novel UX pattern)
2. **Personality-Driven Multi-Agent** - Research team with names, not faceless AI
3. **Spatial Knowledge Organization** - Canvas + AR/VR for intuitive document management
4. **RAPTOR at Scale** - Lazy rebuilding + incremental updates for 1000s of docs
5. **Local-First Privacy** - No data leaves your machine

---

## 📝 Design Principles

- **User as CEO**: Agents are assistants, not autonomous overlords
- **Personality = Interface**: Names and quirks reduce cognitive load
- **Spatial Memory**: Visual organization leverages human spatial reasoning
- **Explicit Limitations**: Agents admit what they can't do
- **Local by Default**: Privacy and speed through edge computing

---

## 🔮 Future Vision

- **Mobile app** with AR mode (scan room, place knowledge orbs)
- **Voice-first** interaction (AirPods + spatial audio)
- **Collaborative mode** (shared workspaces with team agents)
- **Plugin ecosystem** (custom tools, integrations)
- **Enterprise deployment** (on-prem with security auditing)

---

**License:** MIT  
**Status:** Early Development  
**Contact:** [Your info here]