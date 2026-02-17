// ============================================
// DataTable — TanStack Table Wrapper — Strategy §5
// ============================================
// Reusable table with sorting, pagination, optional expandable rows.

"use client";

import { useState, type ReactNode } from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getExpandedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
  type ExpandedState,
  type Row,
} from "@tanstack/react-table";
import { ChevronDown, ChevronRight, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { SkeletonTable } from "./SkeletonTable";
import { cn } from "@/lib/utils";

interface PaginationProps {
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  totalItems?: number;
  perPage?: number;
}

interface DataTableProps<TData> {
  columns: ColumnDef<TData, unknown>[];
  data: TData[];
  pagination?: PaginationProps;
  sorting?: boolean;
  expandable?: boolean;
  renderExpanded?: (row: TData) => ReactNode;
  emptyState?: ReactNode;
  loading?: boolean;
  skeletonRows?: number;
  skeletonColumns?: number;
  className?: string;
  onRowClick?: (row: TData) => void;
  getRowId?: (row: TData) => string;
}

export function DataTable<TData>({
  columns,
  data,
  pagination,
  sorting = false,
  expandable = false,
  renderExpanded,
  emptyState,
  loading = false,
  skeletonRows = 8,
  skeletonColumns,
  className,
  onRowClick,
  getRowId,
}: DataTableProps<TData>) {
  const [sortingState, setSortingState] = useState<SortingState>([]);
  const [expanded, setExpanded] = useState<ExpandedState>({});

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting: sortingState,
      expanded,
    },
    onSortingChange: setSortingState,
    onExpandedChange: setExpanded,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: sorting ? getSortedRowModel() : undefined,
    getExpandedRowModel: expandable ? getExpandedRowModel() : undefined,
    getRowId: getRowId ? (row) => getRowId(row) : undefined,
    enableSorting: sorting,
    enableExpanding: expandable,
  });

  if (loading) {
    return (
      <SkeletonTable
        rows={skeletonRows}
        columns={skeletonColumns ?? columns.length}
        className={className}
      />
    );
  }

  if (data.length === 0 && emptyState) {
    return <>{emptyState}</>;
  }

  return (
    <div className={cn("space-y-4", className)}>
      <div className="rounded-2xl border border-border bg-card overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-border">
                {expandable && (
                  <th className="w-10 px-3 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider" />
                )}
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className={cn(
                      "px-5 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider",
                      header.column.getCanSort() && "cursor-pointer select-none"
                    )}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center gap-1.5">
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                      {sorting && header.column.getCanSort() && (
                        <span className="text-muted-foreground/50">
                          {header.column.getIsSorted() === "asc" ? (
                            <ArrowUp className="h-3.5 w-3.5" />
                          ) : header.column.getIsSorted() === "desc" ? (
                            <ArrowDown className="h-3.5 w-3.5" />
                          ) : (
                            <ArrowUpDown className="h-3.5 w-3.5" />
                          )}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                row={row}
                expandable={expandable}
                renderExpanded={renderExpanded}
                onRowClick={onRowClick}
                columnsCount={columns.length}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pagination && pagination.totalPages > 1 && (
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>
            {pagination.totalItems
              ? `Showing ${(pagination.page - 1) * (pagination.perPage ?? 20) + 1}–${Math.min(pagination.page * (pagination.perPage ?? 20), pagination.totalItems)} of ${pagination.totalItems}`
              : `Page ${pagination.page} of ${pagination.totalPages}`}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => pagination.onPageChange(pagination.page - 1)}
              disabled={pagination.page <= 1}
              className="rounded-xl border border-border px-3 py-1.5 text-sm transition-colors duration-150 hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ← Prev
            </button>
            <button
              onClick={() => pagination.onPageChange(pagination.page + 1)}
              disabled={pagination.page >= pagination.totalPages}
              className="rounded-xl border border-border px-3 py-1.5 text-sm transition-colors duration-150 hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Row component with expansion support ──

function TableRow<TData>({
  row,
  expandable,
  renderExpanded,
  onRowClick,
  columnsCount,
}: {
  row: Row<TData>;
  expandable: boolean;
  renderExpanded?: (row: TData) => ReactNode;
  onRowClick?: (row: TData) => void;
  columnsCount: number;
}) {
  const isExpanded = row.getIsExpanded();

  return (
    <>
      <tr
        className={cn(
          "border-b border-border/50 last:border-b-0 transition-colors duration-150",
          onRowClick && "cursor-pointer hover:bg-muted/50",
          isExpanded && "bg-muted/30"
        )}
        onClick={() => {
          if (onRowClick) onRowClick(row.original);
        }}
      >
        {expandable && (
          <td className="w-10 px-3 py-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                row.toggleExpanded();
              }}
              className="p-0.5 rounded hover:bg-muted transition-colors"
            >
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
          </td>
        )}
        {row.getVisibleCells().map((cell) => (
          <td key={cell.id} className="px-5 py-3" style={{ height: "48px" }}>
            {flexRender(cell.column.columnDef.cell, cell.getContext())}
          </td>
        ))}
      </tr>
      {/* Expanded row */}
      {expandable && isExpanded && renderExpanded && (
        <tr className="bg-muted/20">
          <td colSpan={columnsCount + 1} className="px-5 py-4">
            {renderExpanded(row.original)}
          </td>
        </tr>
      )}
    </>
  );
}
