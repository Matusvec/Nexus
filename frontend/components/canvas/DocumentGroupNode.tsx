"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { motion } from "framer-motion";
import { Folder, MoreHorizontal, Trash2, Edit, Users } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { PERSONAS, type DocumentGroup, type PersonaId } from "@/lib/types";
import { cn } from "@/lib/utils";

interface DocumentGroupNodeProps extends NodeProps {
  data: DocumentGroup;
}

function DocumentGroupNode({ data, selected }: DocumentGroupNodeProps) {
  const persona = data.assignedPersona ? PERSONAS[data.assignedPersona] : null;

  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-primary !border-2 !border-background"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-primary !border-2 !border-background"
      />

      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
      >
        <Card
          className={cn(
            "w-64 cursor-pointer transition-all duration-300 hover:scale-[1.02]",
            selected && "ring-2 ring-primary shadow-lg shadow-primary/25",
            !selected && "hover:shadow-lg"
          )}
          style={{
            borderColor: data.color ? `${data.color}50` : undefined,
          }}
        >
          <CardContent className="p-4">
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: `${data.color}20` }}
                >
                  <Folder className="w-5 h-5" style={{ color: data.color }} />
                </div>
                <div>
                  <h3 className="font-semibold text-sm leading-tight">
                    {data.name}
                  </h3>
                  <p className="text-xs text-muted-foreground">
                    {data.documentIds.length} document
                    {data.documentIds.length !== 1 ? "s" : ""}
                  </p>
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
                    <Edit className="w-4 h-4 mr-2" />
                    Edit Group
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Users className="w-4 h-4 mr-2" />
                    Assign Persona
                  </DropdownMenuItem>
                  <DropdownMenuItem className="text-destructive">
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {/* Description */}
            {data.description && (
              <p className="text-xs text-muted-foreground mb-3 line-clamp-2">
                {data.description}
              </p>
            )}

            {/* Assigned Persona */}
            {persona && (
              <div
                className="flex items-center gap-2 p-2 rounded-lg"
                style={{ backgroundColor: `${persona.color}15` }}
              >
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center text-sm"
                  style={{ backgroundColor: `${persona.color}30` }}
                >
                  {persona.avatar}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium">{persona.name}</p>
                  <p className="text-xs text-muted-foreground truncate">
                    {persona.role}
                  </p>
                </div>
              </div>
            )}

            {/* Document Preview Tags */}
            {data.documentIds.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-3">
                <Badge variant="outline" className="text-xs">
                  {data.documentIds.length} files
                </Badge>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </>
  );
}

export default memo(DocumentGroupNode);
