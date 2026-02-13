# Nexus Frontend

A beautiful, modern frontend for the Nexus AI-powered research document management system.

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn or pnpm

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

---

## 📁 Project Structure

```
frontend/
├── app/                      # Next.js App Router
│   ├── layout.tsx           # Root layout
│   ├── page.tsx             # Landing page
│   ├── globals.css          # Global styles
│   ├── workspace/           # Main workspace with canvas
│   │   └── page.tsx
│   └── documents/           # Document management
│       └── page.tsx
├── components/
│   ├── ui/                  # Reusable UI components (shadcn/ui style)
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── dialog.tsx
│   │   └── ...
│   ├── canvas/              # React Flow canvas components
│   │   ├── WorkspaceCanvas.tsx
│   │   └── DocumentGroupNode.tsx
│   ├── chat/                # AI chat interface
│   │   └── ChatInterface.tsx
│   ├── documents/           # Document-related components
│   │   └── UploadModal.tsx
│   └── layout/              # Layout components
│       ├── Sidebar.tsx
│       └── SearchCommand.tsx
├── lib/
│   ├── api.ts               # API client functions
│   ├── store.ts             # Zustand state management
│   ├── types.ts             # TypeScript type definitions
│   └── utils.ts             # Utility functions
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── API_SPECIFICATION.md     # Backend API docs
```

---

## 🎨 Features

### Landing Page (/)
- Hero section with animated canvas preview
- Feature highlights
- AI Persona cards
- How it works section

### Workspace (/workspace)
- **Infinite Canvas**: Drag-drop document groups, create connections
- **AI Chat Sidebar**: Talk to specialist personas (Max, Elena, Byte, Stacy)
- **Human Task Cards**: View and complete tasks requested by AI
- **Quick Actions**: Upload, search, add groups

### Documents (/documents)
- Grid/List view of all documents
- Upload progress tracking
- Document stats (chunks, size, status)
- Filter and search

---

## 🔌 Backend Integration

The frontend expects a FastAPI backend running on `http://localhost:8000`.

See [API_SPECIFICATION.md](./API_SPECIFICATION.md) for the complete API documentation.

### Quick Start

1. Start the backend:
```bash
cd backend
python main.py  # or uvicorn main:app --reload
```

2. Start the frontend:
```bash
cd frontend
npm run dev
```

---

## 🎭 AI Personas

| Persona | Role | Color | Avatar |
|---------|------|-------|--------|
| Max | Mechanical Engineer | Orange (#F97316) | 🔧 |
| Dr. Elena | Physicist | Purple (#8B5CF6) | ⚛️ |
| Byte | Software Engineer | Green (#10B981) | 💻 |
| Stacy | Electrical Engineer | Blue (#3B82F6) | ⚡ |

---

## 🛠 Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (Radix primitives)
- **Canvas**: React Flow (@xyflow/react)
- **Animations**: Framer Motion
- **State Management**: Zustand
- **Icons**: Lucide React
- **File Upload**: react-dropzone

---

## 📝 Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🎯 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `⌘K` / `Ctrl+K` | Open search |
| `⌘1` | Go to Canvas |
| `⌘2` | Go to Documents |
| `Enter` | Send chat message |
| `Shift+Enter` | New line in chat |

---

## 📄 License

MIT License
