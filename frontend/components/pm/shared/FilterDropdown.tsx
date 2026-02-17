// ============================================
// FilterDropdown — Strategy §5
// ============================================
// Reusable filter dropdown for lists. Wraps Radix Select.

"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface FilterOption {
  value: string;
  label: string;
}

interface FilterDropdownProps {
  label: string;
  value: string;
  options: FilterOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  allLabel?: string;
}

export function FilterDropdown({
  label,
  value,
  options,
  onChange,
  placeholder,
  className,
  allLabel = "All",
}: FilterDropdownProps) {
  return (
    <div className={cn("flex flex-col gap-1", className)}>
      <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-[0.1em]">
        {label}
      </label>
      <Select value={value || "__all__"} onValueChange={(v) => onChange(v === "__all__" ? "" : v)}>
        <SelectTrigger className="h-9 w-[160px] text-sm">
          <SelectValue placeholder={placeholder ?? `${allLabel}`} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="__all__">{allLabel}</SelectItem>
          {options.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
