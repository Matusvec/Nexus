"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  FileText,
  Search,
  Filter,
  Grid,
  List,
  MoreHorizontal,
  Trash2,
  Eye,
  Download,
  Layers,
  Clock,
  HardDrive,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import Link from "next/link";
import Sidebar from "@/components/layout/Sidebar";
import UploadModal from "@/components/documents/UploadModal";
import SearchCommand from "@/components/layout/SearchCommand";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDocumentsStore, useUIStore } from "@/lib/store";
import { formatBytes, formatDate, cn } from "@/lib/utils";
import type { Document } from "@/lib/types";

// Mock data for demonstration
const mockDocuments: Document[] = [
  {
    id: "1",
    filename: "RAPTOR_Paper_ICLR2024.pdf",
    uploadedAt: "2026-01-20T10:30:00Z",
    fileSize: 2456000,
    chunkCount: 45,
    status: "ready",
    summary: "Recursive Abstractive Processing for Tree-Organized Retrieval - A hierarchical approach to RAG systems.",
  },
  {
    id: "2",
    filename: "Motor_Specifications_v2.pdf",
    uploadedAt: "2026-01-22T14:15:00Z",
    fileSize: 1234000,
    chunkCount: 23,
    status: "ready",
    summary: "Technical specifications for brushless DC motors including torque curves and thermal limits.",
  },
  {
    id: "3",
    filename: "Implementation_Notes.md",
    uploadedAt: "2026-01-25T09:00:00Z",
    fileSize: 45000,
    chunkCount: 12,
    status: "ready",
    summary: "Development notes for the Nexus RAG system implementation.",
  },
  {
    id: "4",
    filename: "Quantum_Computing_Intro.pdf",
    uploadedAt: "2026-01-26T16:45:00Z",
    fileSize: 5670000,
    chunkCount: 0,
    status: "processing",
  },
  {
    id: "5",
    filename: "ML_Algorithms_Handbook.pdf",
    uploadedAt: "2026-01-27T11:20:00Z",
    fileSize: 8900000,
    chunkCount: 89,
    status: "ready",
    summary: "Comprehensive guide to machine learning algorithms and their implementations.",
  },
];

function DocumentCard({
  document,
  viewMode,
}: {
  document: Document;
  viewMode: "grid" | "list";
}) {
  const getStatusIcon = () => {
    switch (document.status) {
      case "ready":
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case "processing":
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      case "error":
        return <AlertCircle className="w-4 h-4 text-red-500" />;
    }
  };

  const getFileExtension = (filename: string) => {
    return filename.split(".").pop()?.toUpperCase() || "FILE";
  };

  if (viewMode === "list") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-4 p-4 rounded-lg border border-border hover:border-primary/50 hover:bg-muted/50 transition-all group"
      >
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
          <FileText className="w-5 h-5 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-medium text-sm truncate">{document.filename}</h3>
            <Badge variant="outline" className="text-xs flex-shrink-0">
              {getFileExtension(document.filename)}
            </Badge>
          </div>
          {document.summary && (
            <p className="text-xs text-muted-foreground truncate">
              {document.summary}
            </p>
          )}
        </div>
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Layers className="w-4 h-4" />
            <span>{document.chunkCount} chunks</span>
          </div>
          <div className="flex items-center gap-1">
            <HardDrive className="w-4 h-4" />
            <span>{formatBytes(document.fileSize)}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            <span>{formatDate(document.uploadedAt)}</span>
          </div>
          {getStatusIcon()}
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <MoreHorizontal className="w-4 h-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem>
              <Eye className="w-4 h-4 mr-2" />
              View Details
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Download className="w-4 h-4 mr-2" />
              Download
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-destructive">
              <Trash2 className="w-4 h-4 mr-2" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <Card className="group hover:border-primary/50 hover:shadow-lg transition-all cursor-pointer">
        <CardContent className="p-4">
          <div className="flex items-start justify-between mb-3">
            <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
              <FileText className="w-6 h-6 text-primary" />
            </div>
            <div className="flex items-center gap-2">
              {getStatusIcon()}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <MoreHorizontal className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem>
                    <Eye className="w-4 h-4 mr-2" />
                    View Details
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className="text-destructive">
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          <h3 className="font-medium text-sm mb-1 truncate">{document.filename}</h3>
          <Badge variant="outline" className="text-xs mb-3">
            {getFileExtension(document.filename)}
          </Badge>

          {document.summary && (
            <p className="text-xs text-muted-foreground mb-3 line-clamp-2">
              {document.summary}
            </p>
          )}

          <div className="flex items-center justify-between text-xs text-muted-foreground pt-3 border-t border-border">
            <span className="flex items-center gap-1">
              <Layers className="w-3 h-3" />
              {document.chunkCount} chunks
            </span>
            <span>{formatBytes(document.fileSize)}</span>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

export default function DocumentsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const { setUploadModalOpen } = useUIStore();

  // Use mock data for now
  const documents = mockDocuments;

  const filteredDocuments = documents.filter((doc) =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const stats = {
    total: documents.length,
    ready: documents.filter((d) => d.status === "ready").length,
    processing: documents.filter((d) => d.status === "processing").length,
    totalChunks: documents.reduce((acc, d) => acc + d.chunkCount, 0),
    totalSize: documents.reduce((acc, d) => acc + d.fileSize, 0),
  };

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background">
      {/* Sidebar */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-border flex items-center justify-between px-6">
          <div>
            <h1 className="text-xl font-semibold">Documents</h1>
            <p className="text-sm text-muted-foreground">
              Manage your knowledge base documents
            </p>
          </div>
          <Button onClick={() => setUploadModalOpen(true)}>
            <FileText className="w-4 h-4 mr-2" />
            Upload Document
          </Button>
        </header>

        {/* Stats Bar */}
        <div className="px-6 py-4 border-b border-border bg-muted/30">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <FileText className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats.total}</p>
                <p className="text-xs text-muted-foreground">Documents</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-lg bg-nexus-purple/10 flex items-center justify-center">
                <Layers className="w-5 h-5 text-nexus-purple" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats.totalChunks}</p>
                <p className="text-xs text-muted-foreground">Total Chunks</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-lg bg-nexus-cyan/10 flex items-center justify-center">
                <HardDrive className="w-5 h-5 text-nexus-cyan" />
              </div>
              <div>
                <p className="text-2xl font-bold">{formatBytes(stats.totalSize)}</p>
                <p className="text-xs text-muted-foreground">Total Size</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                <CheckCircle2 className="w-5 h-5 text-green-500" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats.ready}</p>
                <p className="text-xs text-muted-foreground">Ready</p>
              </div>
            </div>
            {stats.processing > 0 && (
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                  <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.processing}</p>
                  <p className="text-xs text-muted-foreground">Processing</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Filters & Search */}
        <div className="px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search documents..."
                className="pl-9"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button variant="outline" size="sm">
              <Filter className="w-4 h-4 mr-2" />
              Filter
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={viewMode === "grid" ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => setViewMode("grid")}
                >
                  <Grid className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Grid View</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={viewMode === "list" ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => setViewMode("list")}
                >
                  <List className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>List View</TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Document List */}
        <div className="flex-1 overflow-auto px-6 pb-6">
          {viewMode === "grid" ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredDocuments.map((doc) => (
                <DocumentCard key={doc.id} document={doc} viewMode="grid" />
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredDocuments.map((doc) => (
                <DocumentCard key={doc.id} document={doc} viewMode="list" />
              ))}
            </div>
          )}

          {filteredDocuments.length === 0 && (
            <div className="text-center py-12">
              <FileText className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No documents found</h3>
              <p className="text-muted-foreground mb-4">
                {searchQuery
                  ? "Try a different search term"
                  : "Upload your first document to get started"}
              </p>
              {!searchQuery && (
                <Button onClick={() => setUploadModalOpen(true)}>
                  Upload Document
                </Button>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Modals */}
      <UploadModal />
      <SearchCommand />
    </div>
  );
}
