"use client";

import { useRef, useState } from "react";

type SelectedFile = {
  file: File;
  id: string;
};

export default function UploadPage() {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  function handleFiles(selected: FileList | null) {
    if (!selected) return;

    const pdfFiles = Array.from(selected).filter(
      (f) => f.type === "application/pdf"
    );

    const mapped = pdfFiles.map((file) => ({
      file,
      id: crypto.randomUUID(),
    }));

    setFiles((prev) => [...prev, ...mapped]);
  }

  function removeFile(id: string) {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }

  function onUploadClick() {
    inputRef.current?.click();
  }

  async function uploadDocuments() {
    if (files.length === 0) {
      alert("Please select at least one PDF file.");
      return;
    }

    // Backend will be connected later
    alert("Upload API will be connected after frontend is completed.");
  }

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-semibold mb-1">
        Upload Legal Documents
      </h1>

      <p className="text-gray-600 mb-6">
        Upload multiple legal PDF documents for RAG-based analysis.
      </p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={onUploadClick}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-10 text-center transition
        ${
          isDragging
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 bg-white"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="application/pdf"
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />

        <p className="text-sm font-medium text-gray-700">
          Drag & drop PDF files here
        </p>

        <p className="text-xs text-gray-500 mt-1">
          or click to select files
        </p>
      </div>

      {/* Selected files */}
      {files.length > 0 && (
        <div className="mt-6">
          <h2 className="text-sm font-semibold mb-2">
            Selected documents
          </h2>

          <ul className="space-y-2">
            {files.map((item) => (
              <li
                key={item.id}
                className="flex items-center justify-between rounded-md border bg-white px-3 py-2 text-sm"
              >
                <span className="truncate">
                  {item.file.name}
                </span>

                <button
                  onClick={() => removeFile(item.id)}
                  className="text-red-500 hover:underline"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Upload button */}
      <div className="mt-6">
        <button
          onClick={uploadDocuments}
          className="rounded-md bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          Upload documents
        </button>
      </div>
    </div>
  );
}
