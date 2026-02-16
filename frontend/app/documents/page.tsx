"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
  FileText,
  Search,
  Grid,
  List,
  MoreHorizontal,
  Trash2,
  Eye,
  Layers,
  Clock,
  AlertCircle,
  Loader2,
  Plus,
  Zap,
} from "lucide-react";
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
import { useEvidenceStore, useUIStore } from "@/lib/store";
import { listEvidence, deleteEvidence, getProblemStats } from "@/lib/api";
import { formatDate, cn } from "@/lib/utils";
import type { Evidence, ProblemStats } from "@/lib/types";

const SOURCE_COLORS: Record<string, string> = {
  interview: "#8B5CF6",
  support_ticket: "#F97316",
  sales_note: "#10B981",
  survey: "#3B82F6",
  other: "#6B7280",
};

function EvidenceCard({
  evidence,
  viewMode,
  onDelete,
}: {
  evidence: Evidence;
  viewMode: "grid" | "list";
  onDelete: (id: string) => void;
}) {
  const color = SOURCE_COLORS[evidence.source_type] || "#6B7280";

  if (viewMode === "list") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-4 p-4 rounded-lg border border-border hover:border-primary/50 hover:bg-muted/50 transition-all group"
      >
        <div
          className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: `${color}20` }}
        >
          <FileText className="w-5 h-5" style={{ color }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-medium text-sm truncate">{evidence.title}</h3>
            <Badge
              variant="outline"
              className="text-xs flex-shrink-0"
              style={{ borderColor: `${color}50`, color }}
            >
              {evidence.source_type.replace("_", " ")}
            </Badge>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {evidence.persona && <span>{evidence.persona}</span>}
            {evidence.segment && <span>{evidence.segment}</span>}
          </div>
        </div>
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Layers className="w-4 h-4" />
            <span>{evidence.chunk_count} chunks</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            <span>
              {evidence.created_at ? formatDate(evidence.created_at) : "—"}
            </span>
          </div>
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
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-destructive"
              onClick={() => onDelete(evidence.id)}
            >
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
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center"
              style={{ backgroundColor: `${color}20` }}
            >
              <FileText className="w-6 h-6" style={{ color }} />
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
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive"
                  onClick={() => onDelete(evidence.id)}
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <h3 className="font-medium text-sm mb-1 truncate">{evidence.title}</h3>
          <Badge
            variant="outline"
            className="text-xs mb-3"
            style={{ borderColor: `${color}50`, color }}
          >
            {evidence.source_type.replace("_", " ")}
          </Badge>

          <div className="flex flex-wrap gap-1 mb-3">
            {evidence.persona && (
              <Badge variant="secondary" className="text-xs">
                {evidence.persona}
              </Badge>
            )}
            {evidence.segment && (
              <Badge variant="secondary" className="text-xs">
                {evidence.segment}
              </Badge>
            )}
          </div>

          <div className="flex items-center justify-between text-xs text-muted-foreground pt-3 border-t border-border">
            <span className="flex items-center gap-1">
              <Layers className="w-3 h-3" />
              {evidence.chunk_count} chunks
            </span>
            <span>
              {evidence.created_at ? formatDate(evidence.created_at) : "—"}
            </span>
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
  const [stats, setStats] = useState<ProblemStats | null>(null);
  const { setUploadModalOpen } = useUIStore();
  const { items, total, isLoading, setItems, setLoading, removeItem } =
    useEvidenceStore();

  const loadEvidence = useCallback(async () => {
    setLoading(true);
    try {
      const res = await listEvidence(1, 50);
      setItems(res.items, res.total);
    } catch (err) {
      console.error("Failed to load evidence:", err);
    } finally {
      setLoading(false);
    }
  }, [setItems, setLoading]);

  const loadStats = useCallback(async () => {
    try {
      const s = await getProblemStats();
      setStats(s);
    } catch {
      // stats may not be available yet
    }
  }, []);

  useEffect(() => {
    loadEvidence();
    loadStats();
  }, [loadEvidence, loadStats]);

  const handleDelete = async (id: string) => {
    try {
      await deleteEvidence(id);
      removeItem(id);
    } catch (err) {
      console.error("Failed to delete evidence:", err);
    }
  };

  const filteredItems = items.filter((e) =>
    e.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const totalChunks = items.reduce((acc, e) => acc + e.chunk_count, 0);

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-border flex items-center justify-between px-6">
          <div>
            <h1 className="text-xl font-semibold">Evidence</h1>
            <p className="text-sm text-muted-foreground">
              Interviews, tickets, and research data
            </p>
          </div>
          <Button onClick={() => setUploadModalOpen(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Add Evidence
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
                <p className="text-2xl font-bold">{total}</p>
                <p className="text-xs text-muted-foreground">Evidence</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-lg bg-nexus-purple/10 flex items-center justify-center">
                <Layers className="w-5 h-5 text-nexus-purple" />
              </div>
              <div>
                <p className="text-2xl font-bold">{totalChunks}</p>
                <p className="text-xs text-muted-foreground">Total Chunks</p>
              </div>
            </div>
            {stats && (
              <>
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-lg bg-nexus-cyan/10 flex items-center justify-center">
                    <Zap className="w-5 h-5 text-nexus-cyan" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{stats.total}</p>
                    <p className="text-xs text-muted-foreground">
                      Problems Found
                    </p>
                  </div>
                </div>
                {(stats.by_severity.critical ?? 0) > 0 && (
                  <div className="flex items-center gap-2">
                    <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
                      <AlertCircle className="w-5 h-5 text-red-500" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold">
                        {stats.by_severity.critical}
                      </p>
                      <p className="text-xs text-muted-foreground">Critical</p>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Filters & Search */}
        <div className="px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search evidence..."
                className="pl-9"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
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

        {/* Evidence List */}
        <div className="flex-1 overflow-auto px-6 pb-6">
          {isLoading ? (
            <div className="text-center py-12">
              <Loader2 className="w-8 h-8 mx-auto text-primary animate-spin mb-4" />
              <p className="text-muted-foreground">Loading evidence...</p>
            </div>
          ) : viewMode === "grid" ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredItems.map((e) => (
                <EvidenceCard
                  key={e.id}
                  evidence={e}
                  viewMode="grid"
                  onDelete={handleDelete}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredItems.map((e) => (
                <EvidenceCard
                  key={e.id}
                  evidence={e}
                  viewMode="list"
                  onDelete={handleDelete}
                />
              ))}
            </div>
          )}

          {!isLoading && filteredItems.length === 0 && (
            <div className="text-center py-12">
              <FileText className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No evidence found</h3>
              <p className="text-muted-foreground mb-4">
                {searchQuery
                  ? "Try a different search term"
                  : "Add your first piece of evidence to get started"}
              </p>
              {!searchQuery && (
                <Button onClick={() => setUploadModalOpen(true)}>
                  <Plus className="w-4 h-4 mr-2" />
                  Add Evidence
                </Button>
              )}
            </div>
          )}
        </div>
      </main>

      <UploadModal />
      <SearchCommand />
    </div>
  );
}
