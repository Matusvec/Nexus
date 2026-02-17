import PageHeader from "@/components/pm/PageHeader";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import { pmFetchSafe } from "@/lib/pm/api";
import type { CostResponse } from "@/lib/pm/types";
import { BarChart3, DollarSign, Cpu, Zap } from "lucide-react";

export default async function UsagePage() {
  const data = await pmFetchSafe<CostResponse>("/llm/costs");

  const models = data ? Object.entries(data.by_model) : [];

  const statCards = data
    ? [
        {
          label: "Total Calls",
          value: data.total_calls.toLocaleString(),
          icon: Zap,
        },
        {
          label: "Input Tokens",
          value: data.total_input_tokens.toLocaleString(),
          icon: Cpu,
        },
        {
          label: "Output Tokens",
          value: data.total_output_tokens.toLocaleString(),
          icon: Cpu,
        },
        {
          label: "Total Cost",
          value: `$${data.total_cost_usd.toFixed(4)}`,
          icon: DollarSign,
        },
      ]
    : [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Usage"
        description="Track LLM cost, token consumption, and processing volume."
      />

      {/* Totals */}
      {statCards.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {statCards.map((stat) => (
            <div
              key={stat.label}
              className="rounded-2xl border border-border bg-card p-5"
            >
              <div className="flex items-center gap-2 text-muted-foreground">
                <stat.icon className="h-4 w-4" strokeWidth={1.75} />
                <span className="text-[11px] font-medium uppercase tracking-[0.1em]">
                  {stat.label}
                </span>
              </div>
              <p className="mt-3 text-2xl font-semibold tabular-nums">
                {stat.value}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Per-model breakdown */}
      {models.length === 0 ? (
        <EmptyState
          icon={BarChart3}
          title="No usage recorded"
          description="Costs and token usage will appear after running extraction, clustering, or proposal generation jobs."
        />
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-card">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-[11px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
              <tr>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3 text-right">Calls</th>
                <th className="px-4 py-3 text-right">Input Tokens</th>
                <th className="px-4 py-3 text-right">Output Tokens</th>
                <th className="px-4 py-3 text-right">Cost</th>
              </tr>
            </thead>
            <tbody>
              {models.map(([model, info]) => (
                <tr
                  key={model}
                  className="border-t border-border transition-colors duration-100 hover:bg-muted/30"
                >
                  <td className="px-4 py-3 font-mono text-xs">{model}</td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {info.calls.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {info.input_tokens.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {info.output_tokens.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums font-medium">
                    ${info.cost_usd.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
