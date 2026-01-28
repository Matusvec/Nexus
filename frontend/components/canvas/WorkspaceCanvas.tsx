"use client";

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge,
  type Node,
  BackgroundVariant,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { motion } from "framer-motion";
import { Plus, Upload, Search, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import DocumentGroupNode from "./DocumentGroupNode";
import { useCanvasStore, useUIStore } from "@/lib/store";
import type { DocumentGroup } from "@/lib/types";
import { generateId } from "@/lib/utils";

// Define node types
const nodeTypes = {
  documentGroup: DocumentGroupNode,
};

// Default edge options
const defaultEdgeOptions = {
  style: { strokeWidth: 2, stroke: "hsl(217 91% 60% / 0.5)" },
  type: "smoothstep",
  animated: true,
};

// Sample data for demo
const initialNodes: Node[] = [
  {
    id: "group-1",
    type: "documentGroup",
    position: { x: 100, y: 100 },
    data: {
      id: "group-1",
      name: "Mechanical Engineering",
      description: "CAD files, motor specs, and material datasheets",
      color: "#F97316",
      documentIds: ["doc-1", "doc-2", "doc-3"],
      position: { x: 100, y: 100 },
      assignedPersona: "max",
    } as DocumentGroup,
  },
  {
    id: "group-2",
    type: "documentGroup",
    position: { x: 450, y: 50 },
    data: {
      id: "group-2",
      name: "Physics Research",
      description: "Electromagnetic theory and quantum mechanics papers",
      color: "#8B5CF6",
      documentIds: ["doc-4", "doc-5"],
      position: { x: 450, y: 50 },
      assignedPersona: "elena",
    } as DocumentGroup,
  },
  {
    id: "group-3",
    type: "documentGroup",
    position: { x: 300, y: 300 },
    data: {
      id: "group-3",
      name: "Software & Algorithms",
      description: "ML papers, code documentation, and system design",
      color: "#10B981",
      documentIds: ["doc-6", "doc-7", "doc-8", "doc-9"],
      position: { x: 300, y: 300 },
      assignedPersona: "byte",
    } as DocumentGroup,
  },
  {
    id: "group-4",
    type: "documentGroup",
    position: { x: 650, y: 250 },
    data: {
      id: "group-4",
      name: "Electronics",
      description: "Circuit designs and PCB layouts",
      color: "#3B82F6",
      documentIds: ["doc-10"],
      position: { x: 650, y: 250 },
      assignedPersona: "stacy",
    } as DocumentGroup,
  },
];

const initialEdges: Edge[] = [
  { id: "e1-2", source: "group-1", target: "group-2", animated: true },
  { id: "e1-3", source: "group-1", target: "group-3" },
  { id: "e3-4", source: "group-3", target: "group-4", animated: true },
];

export default function WorkspaceCanvas() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { setUploadModalOpen, setSearchOpen } = useUIStore();

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) =>
        addEdge({ ...params, animated: true, style: defaultEdgeOptions.style }, eds)
      );
    },
    [setEdges]
  );

  const handleAddGroup = useCallback(() => {
    const newNode: Node = {
      id: `group-${generateId()}`,
      type: "documentGroup",
      position: { x: Math.random() * 400 + 100, y: Math.random() * 300 + 100 },
      data: {
        id: `group-${generateId()}`,
        name: "New Group",
        description: "Click to add documents",
        color: "#6B7280",
        documentIds: [],
        position: { x: 0, y: 0 },
      } as DocumentGroup,
    };
    setNodes((nds) => [...nds, newNode]);
  }, [setNodes]);

  // Stats for the panel
  const stats = useMemo(() => {
    const totalDocs = nodes.reduce(
      (acc, node) => acc + (node.data as DocumentGroup).documentIds.length,
      0
    );
    return {
      groups: nodes.length,
      documents: totalDocs,
      connections: edges.length,
    };
  }, [nodes, edges]);

  return (
    <div className="w-full h-full relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        className="bg-background"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="hsl(217 33% 17%)"
        />
        <Controls className="!bg-card !border-border !shadow-xl" />
        <MiniMap
          nodeColor={(node) => {
            const data = node.data as DocumentGroup;
            return data.color || "#6B7280";
          }}
          maskColor="hsl(222 47% 5% / 0.8)"
          className="!bg-card !border !border-border"
        />

        {/* Top Left Panel - Stats */}
        <Panel position="top-left">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass rounded-xl p-4 space-y-2"
          >
            <h3 className="font-semibold text-sm mb-3">Workspace Overview</h3>
            <div className="flex gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-primary">{stats.groups}</p>
                <p className="text-xs text-muted-foreground">Groups</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-nexus-purple">
                  {stats.documents}
                </p>
                <p className="text-xs text-muted-foreground">Documents</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-nexus-cyan">
                  {stats.connections}
                </p>
                <p className="text-xs text-muted-foreground">Links</p>
              </div>
            </div>
          </motion.div>
        </Panel>

        {/* Top Right Panel - Actions */}
        <Panel position="top-right">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex gap-2"
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  className="glass"
                  onClick={() => setSearchOpen(true)}
                >
                  <Search className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Search Documents (⌘K)</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  className="glass"
                  onClick={() => setUploadModalOpen(true)}
                >
                  <Upload className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Upload Document</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  className="glass"
                  onClick={handleAddGroup}
                >
                  <Plus className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Add Group</TooltipContent>
            </Tooltip>
          </motion.div>
        </Panel>

        {/* Bottom Panel - RAPTOR Info */}
        <Panel position="bottom-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Badge variant="outline" className="glass px-4 py-2">
              <Layers className="w-3 h-3 mr-2" />
              RAPTOR Tree: 3 Layers • Collapsed Tree Retrieval Active
            </Badge>
          </motion.div>
        </Panel>
      </ReactFlow>
    </div>
  );
}
