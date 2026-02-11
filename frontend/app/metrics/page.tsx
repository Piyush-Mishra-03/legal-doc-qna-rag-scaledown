"use client";

import { useEffect, useState } from "react";

type Metrics = {
  totalDocuments: number;
  totalChunks: number;
  avgOriginalTokens: number;
  avgCompressedTokens: number;
  avgLatencyMs: number;
};

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  useEffect(() => {
    async function load() {
      /*
        This is a mock.
        It matches the FINAL backend contract.
      */

      await new Promise((r) => setTimeout(r, 400));

      const mock: Metrics = {
        totalDocuments: 38,
        totalChunks: 12460,
        avgOriginalTokens: 820,
        avgCompressedTokens: 310,
        avgLatencyMs: 740,
      };

      setMetrics(mock);
    }

    load();
  }, []);

  const reduction =
    metrics
      ? Math.round(
          (1 -
            metrics.avgCompressedTokens /
              metrics.avgOriginalTokens) *
            100
        )
      : 0;

  return (
    <div className="max-w-5xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold mb-1">
          Compression & Performance Metrics
        </h1>
        <p className="text-gray-600">
          ScaleDown compression impact and system performance.
        </p>
      </div>

      {!metrics && (
        <div className="text-sm text-gray-500">
          Loading metrics...
        </div>
      )}

      {metrics && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">

            <Card
              title="Documents"
              value={metrics.totalDocuments.toString()}
            />

            <Card
              title="Chunks"
              value={metrics.totalChunks.toString()}
            />

            <Card
              title="Avg latency"
              value={`${metrics.avgLatencyMs} ms`}
            />

            <Card
              title="Token reduction"
              value={`${reduction}%`}
            />

          </div>

          <div className="rounded-lg border bg-white p-5 space-y-3">
            <h2 className="text-sm font-semibold">
              Token statistics (average per chunk)
            </h2>

            <div className="flex justify-between text-sm">
              <span>Original tokens</span>
              <span>{metrics.avgOriginalTokens}</span>
            </div>

            <div className="flex justify-between text-sm">
              <span>Compressed tokens</span>
              <span>{metrics.avgCompressedTokens}</span>
            </div>

            <div className="mt-3 h-2 rounded bg-gray-200">
              <div
                className="h-2 rounded bg-blue-600"
                style={{
                  width: `${reduction}%`,
                }}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Card({
  title,
  value,
}: {
  title: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border bg-white p-4">
      <div className="text-sm text-gray-500">
        {title}
      </div>
      <div className="mt-1 text-2xl font-semibold">
        {value}
      </div>
    </div>
  );
}
