import PageHeader from "@/components/pm/PageHeader";
import { pmFetchSafe } from "@/lib/pm/api";

interface CostResponse {
  total_calls: number;
  total_cost_usd: number;
  total_input_tokens: number;
  total_output_tokens: number;
  by_model: Record<
    string,
    {
      calls: number;
      input_tokens: number;
      output_tokens: number;
      cost_usd: number;
    }
  >;
}

export default async function UsagePage() {
  const data = await pmFetchSafe<CostResponse>("/llm/costs");

  const models = data ? Object.entries(data.by_model) : [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Usage"
        description="Track LLM cost, job history, and processing volume."
      />

      {/* Totals */}
      {data && (
        <div className="grid gap-4 sm:grid-cols-4">
          {[
            { label: "Total Calls", value: data.total_calls },
            { label: "Input Tokens", value: data.total_input_tokens.toLocaleString() },
            { label: "Output Tokens", value: data.total_output_tokens.toLocaleString() },
            { label: "Total Cost", value: `$${data.total_cost_usd.toFixed(4)}` },
          ].map((stat) => (
            <div key={stat.label} className="rounded-2xl border border-border bg-card/70 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{stat.label}</p>
              <p className="mt-1 text-2xl font-semibold">{stat.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Per-model breakdown */}
      {models.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
          No LLM usage recorded yet. Costs will appear after running extraction or proposal jobs.
        </div>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-card/70">
          <table className="w-full text-sm">
            <thead className="bg-muted/70 text-left text-xs uppercase tracking-[0.2em] text-muted-foreground">
              <tr>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Calls</th>
                <th className="px-4 py-3">Input Tokens</th>
                <th className="px-4 py-3">Output Tokens</th>
                <th className="px-4 py-3">Cost</th>
              </tr>
            </thead>
            <tbody>
              {models.map(([model, info]) => (
                <tr key={model} className="border-t border-border">
                  <td className="px-4 py-3 font-mono text-xs">{model}</td>
                  <td className="px-4 py-3">{info.calls}</td>
                  <td className="px-4 py-3">{info.input_tokens.toLocaleString()}</td>
                  <td className="px-4 py-3">{info.output_tokens.toLocaleString()}</td>
                  <td className="px-4 py-3">${info.cost_usd.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
