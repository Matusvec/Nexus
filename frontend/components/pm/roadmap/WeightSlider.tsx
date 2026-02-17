// ============================================
// WeightSlider — Strategy §4.12
// ============================================
// Strategic weight adjuster using Radix Slider.
// 0.1 – 3.0, step 0.1. Persists on release.

"use client";

import { useState } from "react";
import { Slider } from "@/components/ui/slider";
import { updateWeight } from "@/lib/pm/api";
import { toast } from "sonner";

interface WeightSliderProps {
  proposalId: string;
  initialWeight: number;
  onWeightChange?: (weight: number) => void;
}

export function WeightSlider({
  proposalId,
  initialWeight,
  onWeightChange,
}: WeightSliderProps) {
  const [value, setValue] = useState(initialWeight);

  const handleRelease = async (newValue: number[]) => {
    const weight = newValue[0];
    try {
      await updateWeight(proposalId, weight);
      onWeightChange?.(weight);
    } catch {
      toast.error("Failed to update weight.");
      setValue(initialWeight);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <Slider
        value={[value]}
        onValueChange={(v) => {
          setValue(v[0]);
          onWeightChange?.(v[0]);
        }}
        onValueCommit={handleRelease}
        min={0.1}
        max={3.0}
        step={0.1}
        className="w-32"
      />
      <span className="w-8 text-sm font-semibold text-foreground text-right">
        {value.toFixed(1)}
      </span>
    </div>
  );
}
