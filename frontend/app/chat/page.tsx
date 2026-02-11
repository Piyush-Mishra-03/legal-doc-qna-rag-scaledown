"use client";

import { useState } from "react";

type Citation = {
  fileName: string;
  page: number;
};

type PrecedentLink = {
  title: string;
  similarity: number;
};

type ChatAnswer = {
  answer: string;
  confidence: number;
  citations: Citation[];
  precedents: PrecedentLink[];
};

export default function ChatPage() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ChatAnswer | null>(null);

  async function askQuestion() {
    if (!question.trim()) return;

    setLoading(true);

    /*
      This is a temporary mock.
      It matches the FINAL backend response contract.
    */

    await new Promise((r) => setTimeout(r, 800));

    const mockResponse: ChatAnswer = {
      answer:
        "Based on the uploaded documents, a contract becomes voidable when consent is obtained through coercion, undue influence, fraud or misrepresentation.",
      confidence: 0.86,
      citations: [
        { fileName: "Contract_Act_1872.pdf", page: 14 },
        { fileName: "Case_Law_Alpha.pdf", page: 7 },
      ],
      precedents: [
        { title: "Ranganayakamma vs Alwar Setti (1889)", similarity: 0.91 },
        { title: "Chikkam Ammiraju vs Chikkam Seshamma (1917)", similarity: 0.88 },
      ],
    };

    setResult(mockResponse);
    setLoading(false);
  }

  return (
    <div className="max-w-4xl flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold mb-1">
          Legal Document Chat
        </h1>
        <p className="text-gray-600">
          Ask questions across all uploaded legal documents.
        </p>
      </div>

      {/* Question box */}
      <div className="flex gap-3">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a legal question..."
          className="flex-1 rounded-md border px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        <button
          onClick={askQuestion}
          disabled={loading}
          className="rounded-md bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Asking..." : "Ask"}
        </button>
      </div>

      {/* Result */}
      {result && (
        <div className="rounded-lg border bg-white p-5 space-y-4">

          {/* Answer */}
          <div>
            <h2 className="text-sm font-semibold mb-1">Answer</h2>
            <p className="text-sm text-gray-800 leading-relaxed">
              {result.answer}
            </p>
          </div>

          {/* Confidence */}
          <div>
            <h2 className="text-sm font-semibold mb-1">Confidence</h2>
            <div className="flex items-center gap-2">
              <div className="h-2 w-full rounded bg-gray-200">
                <div
                  className="h-2 rounded bg-green-500"
                  style={{
                    width: `${Math.round(result.confidence * 100)}%`,
                  }}
                />
              </div>
              <span className="text-sm font-medium">
                {Math.round(result.confidence * 100)}%
              </span>
            </div>
          </div>

          {/* Citations */}
          <div>
            <h2 className="text-sm font-semibold mb-1">Citations</h2>
            <ul className="list-disc pl-5 text-sm text-gray-700">
              {result.citations.map((c, idx) => (
                <li key={idx}>
                  {c.fileName} – page {c.page}
                </li>
              ))}
            </ul>
          </div>

          {/* Precedents */}
          <div>
            <h2 className="text-sm font-semibold mb-1">
              Related precedents
            </h2>

            <ul className="space-y-1 text-sm text-gray-700">
              {result.precedents.map((p, idx) => (
                <li
                  key={idx}
                  className="flex items-center justify-between"
                >
                  <span>{p.title}</span>
                  <span className="text-xs text-gray-500">
                    {(p.similarity * 100).toFixed(0)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>

        </div>
      )}
    </div>
  );
}
