"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  FileText,
  Layers,
  Sparkles,
  Clock,
  BarChart3,
  BookOpen,
  ArrowRight,
  Info,
  Loader2,
  X,
} from "lucide-react";
import Sidebar from "@/components/layout/Sidebar";
import UploadModal from "@/components/documents/UploadModal";
import SearchCommand from "@/components/layout/SearchCommand";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

// Mock retrieval results
interface RetrievalResult {
  id: string;
  content: string;
  documentName: string;
  documentId: string;
  layer: number;
  relevanceScore: number;
  chunkId: string;
  highlights: string[];
}

const mockResults: RetrievalResult[] = [
  {
    id: "r1",
    content:
      "RAPTOR builds a recursive tree structure by clustering document chunks and generating summaries at multiple layers. Layer 0 contains the original text chunks, while higher layers contain progressively more abstract summaries.",
    documentName: "RAPTOR_Paper_ICLR2024.pdf",
    documentId: "doc-1",
    layer: 0,
    relevanceScore: 0.95,
    chunkId: "chunk-1a",
    highlights: ["recursive tree structure", "clustering document chunks", "multiple layers"],
  },
  {
    id: "r2",
    content:
      "The collapsed tree retrieval method outperforms traditional approaches by allowing the model to select from all layers simultaneously. This enables both fine-grained detail retrieval and high-level summary access in a single query.",
    documentName: "RAPTOR_Paper_ICLR2024.pdf",
    documentId: "doc-1",
    layer: 1,
    relevanceScore: 0.91,
    chunkId: "chunk-2b",
    highlights: ["collapsed tree retrieval", "all layers simultaneously"],
  },
  {
    id: "r3",
    content:
      "Brushless DC motors can sustain 5G acceleration loads when properly mounted with 7075 aluminum brackets. The safety margin at rated torque exceeds 3x for standard operating conditions.",
    documentName: "Motor_Specifications_v2.pdf",
    documentId: "doc-2",
    layer: 0,
    relevanceScore: 0.78,
    chunkId: "chunk-3c",
    highlights: ["5G acceleration", "7075 aluminum", "safety margin"],
  },
  {
    id: "r4",
    content:
      "Implementation of the embedding pipeline uses sentence-transformers with the all-MiniLM-L6-v2 model for generating chunk embeddings. ChromaDB is used as the vector store backend for fast similarity search.",
    documentName: "Implementation_Notes.md",
    documentId: "doc-3",
    layer: 0,
    relevanceScore: 0.72,
    chunkId: "chunk-4d",
    highlights: ["sentence-transformers", "ChromaDB", "similarity search"],
  },
];

function ResultCard({
  result,
  isSelected,
  onClick,
}: {
  result: RetrievalResult;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.005 }}
      onClick={onClick}
      className={cn(
        "p-4 rounded-xl border cursor-pointer transition-all",
        isSelected
          ? "border-primary bg-primary/5 shadow-lg shadow-primary/10"
          : "border-border hover:border-primary/30 hover:bg-muted/30"
      )}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <FileText className="w-4 h-4 text-primary" />
          </div>
          <div>
            <p className="text-sm font-medium">{result.documentName}</p>
            <div className="flex items-center gap-2 mt-0.5">
              <Badge variant="info" className="text-xs">
                Layer {result.layer}
              </Badge>
              <span className="text-xs text-muted-foreground">
                Chunk {result.chunkId}
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <BarChart3 className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-sm font-medium text-primary">
            {Math.round(result.relevanceScore * 100)}%
          </span>
        </div>
      </div>

      <p className="text-sm text-muted-foreground leading-relaxed mt-3">
        {result.content}
      </p>

      {result.highlights.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-3">
          {result.highlights.map((h) => (
            <Badge key={h} variant="outline" className="text-xs">
              {h}
            </Badge>
          ))}
        </div>
      )}
    </motion.div>
  );
}

function ResultSkeleton() {
  return (
    <div className="p-4 rounded-xl border border-border">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Skeleton className="w-8 h-8 rounded-lg" />
          <div>
            <Skeleton className="h-4 w-40 mb-1" />
            <Skeleton className="h-3 w-24" />
          </div>
        </div>
        <Skeleton className="h-4 w-12" />
      </div>
      <Skeleton className="h-4 w-full mb-2" />
      <Skeleton className="h-4 w-3/4" />
      <div className="flex gap-2 mt-3">
        <Skeleton className="h-5 w-20 rounded-full" />
        <Skeleton className="h-5 w-16 rounded-full" />
      </div>
    </div>
  );
}

function DetailPanel({ result }: { result: RetrievalResult }) {
  const layerDescriptions: Record<number, string> = {
    0: "Original document chunk — exact text from the source",
    1: "First-level summary — AI-generated overview of clustered chunks",
    2: "High-level theme — abstract summary across multiple clusters",
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="h-full flex flex-col"
    >
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-6">
          {/* Source */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
              Source Document
            </p>
            <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
              <FileText className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">{result.documentName}</span>
            </div>
          </div>

          {/* Retrieval Layer */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
              RAPTOR Layer
            </p>
            <div className="p-3 rounded-lg bg-muted/50">
              <div className="flex items-center gap-2 mb-1">
                <Layers className="w-4 h-4 text-nexus-purple" />
                <span className="text-sm font-medium">Layer {result.layer}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                {layerDescriptions[result.layer] || "AI-generated summary layer"}
              </p>
            </div>
          </div>

          {/* Relevance Score */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
              Relevance Score
            </p>
            <div className="p-3 rounded-lg bg-muted/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">
                  {Math.round(result.relevanceScore * 100)}% match
                </span>
                <BarChart3 className="w-4 h-4 text-primary" />
              </div>
              <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.relevanceScore * 100}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className="h-full bg-primary rounded-full"
                />
              </div>
            </div>
          </div>

          {/* Key Matches */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
              Key Matches
            </p>
            <div className="flex flex-wrap gap-1.5">
              {result.highlights.map((h) => (
                <Badge key={h} variant="info" className="text-xs">
                  {h}
                </Badge>
              ))}
            </div>
          </div>

          {/* Full Content */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
              Full Content
            </p>
            <div className="p-3 rounded-lg bg-muted/50 text-sm leading-relaxed">
              {result.content}
            </div>
          </div>
        </div>
      </ScrollArea>
    </motion.div>
  );
}

export default function RetrievalPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<RetrievalResult | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [activeLayer, setActiveLayer] = useState<"all" | "0" | "1" | "2">("all");

  const handleSearch = () => {
    if (!query.trim()) return;
    setIsSearching(true);
    setHasSearched(true);
    setSelectedResult(null);

    // Simulate search delay
    setTimeout(() => {
      setResults(mockResults);
      setIsSearching(false);
    }, 1200);
  };

  const filteredResults =
    activeLayer === "all"
      ? results
      : results.filter((r) => r.layer === parseInt(activeLayer));

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
            <h1 className="text-xl font-semibold">Retrieval</h1>
            <p className="text-sm text-muted-foreground">
              Query your knowledge base with RAPTOR hierarchical retrieval
            </p>
          </div>
          <Badge variant="outline" className="text-xs">
            <Sparkles className="w-3 h-3 mr-1" />
            RAPTOR Active
          </Badge>
        </header>

        {/* Search Bar */}
        <div className="px-6 py-5 border-b border-border bg-muted/20">
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <Input
                placeholder='Ask a question... e.g. "How does RAPTOR build its tree structure?"'
                className="pl-12 pr-24 h-12 text-base rounded-xl bg-card border-border"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              />
              <Button
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg"
                size="sm"
                onClick={handleSearch}
                disabled={!query.trim() || isSearching}
              >
                {isSearching ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <>
                    Search
                    <ArrowRight className="w-4 h-4 ml-1" />
                  </>
                )}
              </Button>
            </div>

            {/* Quick filters */}
            <div className="flex items-center gap-2 mt-3">
              <span className="text-xs text-muted-foreground">Quick:</span>
              {[
                "Summarize all documents",
                "What is RAPTOR?",
                "Motor specifications",
              ].map((suggestion) => (
                <Button
                  key={suggestion}
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 rounded-full"
                  onClick={() => {
                    setQuery(suggestion);
                  }}
                >
                  {suggestion}
                </Button>
              ))}
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Results List */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Layer Tabs */}
            {hasSearched && (
              <div className="px-6 py-3 border-b border-border flex items-center justify-between">
                <Tabs
                  value={activeLayer}
                  onValueChange={(v) => setActiveLayer(v as typeof activeLayer)}
                >
                  <TabsList>
                    <TabsTrigger value="all">
                      All Layers ({results.length})
                    </TabsTrigger>
                    <TabsTrigger value="0">
                      Detail ({results.filter((r) => r.layer === 0).length})
                    </TabsTrigger>
                    <TabsTrigger value="1">
                      Summary ({results.filter((r) => r.layer === 1).length})
                    </TabsTrigger>
                    <TabsTrigger value="2">
                      Theme ({results.filter((r) => r.layer === 2).length})
                    </TabsTrigger>
                  </TabsList>
                </Tabs>
                <span className="text-xs text-muted-foreground">
                  {filteredResults.length} result{filteredResults.length !== 1 ? "s" : ""}
                </span>
              </div>
            )}

            <ScrollArea className="flex-1 p-6">
              {/* Empty state - no search yet */}
              {!hasSearched && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-center py-20"
                >
                  <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                    <Search className="w-10 h-10 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">
                    Search your knowledge base
                  </h3>
                  <p className="text-muted-foreground max-w-md mx-auto mb-6">
                    Ask a question in natural language. RAPTOR will search across all
                    document layers to find the most relevant answers with citations.
                  </p>
                  <div className="flex items-center justify-center gap-6 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <Layers className="w-4 h-4 text-nexus-purple" />
                      <span>Multi-layer retrieval</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-nexus-cyan" />
                      <span>Source citations</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-nexus-orange" />
                      <span>AI-powered ranking</span>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Loading skeletons */}
              {isSearching && (
                <div className="space-y-4 max-w-3xl">
                  <ResultSkeleton />
                  <ResultSkeleton />
                  <ResultSkeleton />
                </div>
              )}

              {/* Results */}
              {!isSearching && hasSearched && (
                <div className="space-y-3 max-w-3xl">
                  <AnimatePresence>
                    {filteredResults.map((result) => (
                      <ResultCard
                        key={result.id}
                        result={result}
                        isSelected={selectedResult?.id === result.id}
                        onClick={() => setSelectedResult(result)}
                      />
                    ))}
                  </AnimatePresence>

                  {filteredResults.length === 0 && (
                    <div className="text-center py-12">
                      <Info className="w-10 h-10 mx-auto text-muted-foreground mb-3" />
                      <h3 className="font-medium mb-1">
                        No results at this layer
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Try selecting &ldquo;All Layers&rdquo; or adjusting your query
                      </p>
                    </div>
                  )}
                </div>
              )}
            </ScrollArea>
          </div>

          {/* Detail Panel */}
          <AnimatePresence>
            {selectedResult && (
              <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 360, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="border-l border-border overflow-hidden bg-card"
              >
                <div className="flex items-center justify-between p-4 border-b border-border">
                  <span className="text-sm font-medium">Why this result</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => setSelectedResult(null)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
                <DetailPanel result={selectedResult} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <UploadModal />
      <SearchCommand />
    </div>
  );
}
