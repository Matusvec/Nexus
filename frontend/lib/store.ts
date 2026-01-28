import { create } from "zustand";
import type {
  Document,
  DocumentGroup,
  Message,
  HumanTask,
  PersonaId,
  CanvasNode,
  CanvasEdge,
} from "@/lib/types";

// ============================================
// DOCUMENTS STORE
// ============================================
interface DocumentsState {
  documents: Document[];
  selectedDocumentId: string | null;
  isLoading: boolean;
  setDocuments: (documents: Document[]) => void;
  addDocument: (document: Document) => void;
  removeDocument: (id: string) => void;
  selectDocument: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
}

export const useDocumentsStore = create<DocumentsState>((set) => ({
  documents: [],
  selectedDocumentId: null,
  isLoading: false,
  setDocuments: (documents) => set({ documents }),
  addDocument: (document) =>
    set((state) => ({ documents: [...state.documents, document] })),
  removeDocument: (id) =>
    set((state) => ({
      documents: state.documents.filter((d) => d.id !== id),
      selectedDocumentId:
        state.selectedDocumentId === id ? null : state.selectedDocumentId,
    })),
  selectDocument: (id) => set({ selectedDocumentId: id }),
  setLoading: (loading) => set({ isLoading: loading }),
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
      groups: state.groups.map((g) => (g.id === id ? { ...g, ...updates } : g)),
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
  conversationId: string | null;
  isSidebarOpen: boolean;
  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  clearMessages: () => void;
  setActivePersona: (id: PersonaId) => void;
  setStreaming: (streaming: boolean) => void;
  setConversationId: (id: string | null) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  activePersonaId: "max",
  isStreaming: false,
  conversationId: null,
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
  setConversationId: (id) => set({ conversationId: id }),
  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
  setSidebarOpen: (open) => set({ isSidebarOpen: open }),
}));

// ============================================
// TASKS STORE
// ============================================
interface TasksState {
  tasks: HumanTask[];
  activeTaskId: string | null;
  setTasks: (tasks: HumanTask[]) => void;
  addTask: (task: HumanTask) => void;
  updateTask: (id: string, updates: Partial<HumanTask>) => void;
  removeTask: (id: string) => void;
  setActiveTask: (id: string | null) => void;
}

export const useTasksStore = create<TasksState>((set) => ({
  tasks: [],
  activeTaskId: null,
  setTasks: (tasks) => set({ tasks }),
  addTask: (task) => set((state) => ({ tasks: [...state.tasks, task] })),
  updateTask: (id, updates) =>
    set((state) => ({
      tasks: state.tasks.map((t) => (t.id === id ? { ...t, ...updates } : t)),
    })),
  removeTask: (id) =>
    set((state) => ({
      tasks: state.tasks.filter((t) => t.id !== id),
      activeTaskId: state.activeTaskId === id ? null : state.activeTaskId,
    })),
  setActiveTask: (id) => set({ activeTaskId: id }),
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
