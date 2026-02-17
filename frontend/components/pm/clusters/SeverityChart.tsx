// ============================================
// SeverityChart — Strategy §4.8
// ============================================
// Horizontal bar chart for severity distribution using Recharts.

"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface SeverityChartProps {
  distribution: Record<string, number>;
}

const severityColors: Record<string, string> = {
  critical: "#DC2626",
  high: "#EA580C",
  medium: "#D97706",
  low: "#16A34A",
};

const severityOrder = ["critical", "high", "medium", "low"];

export function SeverityChart({ distribution }: SeverityChartProps) {
  const data = severityOrder.map((sev) => ({
    name: sev.charAt(0).toUpperCase() + sev.slice(1),
    key: sev,
    count: distribution[sev] ?? 0,
  }));

  return (
    <div className="h-[140px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 60 }}>
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
            width={60}
            axisLine={false}
            tickLine={false}
          />
          <Bar
            dataKey="count"
            radius={[0, 4, 4, 0]}
            barSize={14}
            label={{
              position: "right",
              fill: "hsl(var(--foreground))",
              fontSize: 12,
              fontWeight: 500,
            }}
          >
            {data.map((entry) => (
              <Cell key={entry.key} fill={severityColors[entry.key]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
