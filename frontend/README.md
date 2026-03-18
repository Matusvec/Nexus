# Nexus Frontend

Next.js frontend for the Nexus PM pipeline — an AI-powered product management tool for generating and managing proposals, tasks, and project artifacts.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (Radix primitives)
- **Animations**: Framer Motion
- **State Management**: Zustand
- **Icons**: Lucide React

## Getting Started

```bash
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
frontend/
├── app/                  # Next.js App Router pages
├── components/
│   ├── ui/               # Reusable UI primitives
│   └── pm/               # PM pipeline feature components
├── lib/                  # API client, types, utilities
└── public/
```

## Backend Integration

Expects a FastAPI backend running on `http://localhost:8000`. Set the URL via environment variable:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the backend:

```bash
cd ../backend
uvicorn main:app --reload
```

## Build

```bash
pnpm build
pnpm start
```
