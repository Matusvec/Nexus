import { create } from "zustand";
import type {
  Evidence,
  ProblemMention,
  ProblemCluster,
  DocumentGroup,
  Message,
  PersonaId,
  CanvasNode,
  CanvasEdge,
  JobStatusResponse,
} from "@/lib/types";

// ============================================
// EVIDENCE STORE (replaces documents)
// ============================================
interface EvidenceState {
  items: Evidence[];
  selectedId: string | null;
  isLoading: boolean;
  total: number;
  page: number;
  setItems: (items: Evidence[], total?: number) => void;
  addItem: (item: Evidence) => void;
  removeItem: (id: string) => void;
  select: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
  setPage: (page: number) => void;
}

export const useEvidenceStore = create<EvidenceState>((set) => ({
  items: [],
  selectedId: null,
  isLoading: false,
  total: 0,
  page: 1,
  setItems: (items, total) =>
    set({ items, ...(total !== undefined ? { total } : {}) }),
  addItem: (item) =>
    set((state) => ({ items: [item, ...state.items], total: state.total + 1 })),
  removeItem: (id) =>
    set((state) => ({
      items: state.items.filter((e) => e.id !== id),
      total: state.total - 1,
      selectedId: state.selectedId === id ? null : state.selectedId,
    })),
  select: (id) => set({ selectedId: id }),
  setLoading: (loading) => set({ isLoading: loading }),
  setPage: (page) => set({ page }),
}));

// ============================================
// PROBLEMS STORE
// ============================================
interface ProblemsState {
  items: ProblemMention[];
  total: number;
  page: number;
  isLoading: boolean;
  setItems: (items: ProblemMention[], total?: number) => void;
  setLoading: (loading: boolean) => void;
  setPage: (page: number) => void;
}

export const useProblemsStore = create<ProblemsState>((set) => ({
  items: [],
  total: 0,
  page: 1,
  isLoading: false,
  setItems: (items, total) =>
    set({ items, ...(total !== undefined ? { total } : {}) }),
  setLoading: (loading) => set({ isLoading: loading }),
  setPage: (page) => set({ page }),
}));

// ============================================
// JOBS STORE
// ============================================
interface JobsState {
  activeJobs: Map<string, JobStatusResponse>;
  setJob: (jobId: string, status: JobStatusResponse) => void;
  removeJob: (jobId: string) => void;
  clearJobs: () => void;
}

export const useJobsStore = create<JobsState>((set) => ({
  activeJobs: new Map(),
  setJob: (jobId, status) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      next.set(jobId, status);
      return { activeJobs: next };
    }),
  removeJob: (jobId) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      next.delete(jobId);
      return { activeJobs: next };
    }),
  clearJobs: () => set({ activeJobs: new Map() }),
}));

// ============================================
// CANVAS STORE
// ============================================
interface CanvasState {
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  selectedNodeId: string | null;
  groups: DocumentGroup[];
  setNodes: (nodes: CanvasNode[]) => void;
  setEdges: (edges: CanvasEdge[]) => void;
  addNode: (node: CanvasNode) => void;
  updateNode: (id: string, data: Partial<DocumentGroup>) => void;
  removeNode: (id: string) => void;
  addEdge: (edge: CanvasEdge) => void;
  removeEdge: (id: string) => void;
  selectNode: (id: string | null) => void;
  setGroups: (groups: DocumentGroup[]) => void;
  addGroup: (group: DocumentGroup) => void;
  updateGroup: (id: string, updates: Partial<DocumentGroup>) => void;
  removeGroup: (id: string) => void;
}

export const useCanvasStore = create<CanvasState>((set) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  groups: [],
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  addNode: (node) => set((state) => ({ nodes: [...state.nodes, node] })),
  updateNode: (id, data) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n
      ),
    })),
  removeNode: (id) =>
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== id),
      edges: state.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
    })),
  addEdge: (edge) => set((state) => ({ edges: [...state.edges, edge] })),
  removeEdge: (id) =>
    set((state) => ({ edges: state.edges.filter((e) => e.id !== id) })),
  selectNode: (id) => set({ selectedNodeId: id }),
  setGroups: (groups) => set({ groups }),
  addGroup: (group) => set((state) => ({ groups: [...state.groups, group] })),
  updateGroup: (id, updates) =>
    set((state) => ({
      groups: state.groups.map((g) =>
        g.id === id ? { ...g, ...updates } : g
      ),
    })),
  removeGroup: (id) =>
    set((state) => ({ groups: state.groups.filter((g) => g.id !== id) })),
}));

// ============================================
// CHAT STORE
// ============================================
interface ChatState {
  messages: Message[];
  activePersonaId: PersonaId;
  isStreaming: boolean;
  isSidebarOpen: boolean;
  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  clearMessages: () => void;
  setActivePersona: (id: PersonaId) => void;
  setStreaming: (streaming: boolean) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  activePersonaId: "max",
  isStreaming: false,
  isSidebarOpen: true,
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  updateMessage: (id, updates) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === id ? { ...m, ...updates } : m
      ),
    })),
  clearMessages: () => set({ messages: [] }),
  setActivePersona: (id) => set({ activePersonaId: id }),
  setStreaming: (streaming) => set({ isStreaming: streaming }),
  toggleSidebar: () =>
    set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
  setSidebarOpen: (open) => set({ isSidebarOpen: open }),
}));

// ============================================
// UI STORE
// ============================================
interface UIState {
  isUploadModalOpen: boolean;
  isSettingsModalOpen: boolean;
  isSearchOpen: boolean;
  activeView: "canvas" | "documents" | "chat";
  setUploadModalOpen: (open: boolean) => void;
  setSettingsModalOpen: (open: boolean) => void;
  setSearchOpen: (open: boolean) => void;
  setActiveView: (view: "canvas" | "documents" | "chat") => void;
}

export const useUIStore = create<UIState>((set) => ({
  isUploadModalOpen: false,
  isSettingsModalOpen: false,
  isSearchOpen: false,
  activeView: "canvas",
  setUploadModalOpen: (open) => set({ isUploadModalOpen: open }),
  setSettingsModalOpen: (open) => set({ isSettingsModalOpen: open }),
  setSearchOpen: (open) => set({ isSearchOpen: open }),
  setActiveView: (view) => set({ activeView: view }),
}));
