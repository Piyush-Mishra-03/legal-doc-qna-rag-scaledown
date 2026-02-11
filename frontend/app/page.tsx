"use client";

import { useState, useRef, useEffect } from 'react';

// Types
interface Message {
  role: 'user' | 'assistant';
  content: string;
  confidence?: number;
  citations?: Citation[];
  timestamp: Date;
}

interface Citation {
  fileName: string;
  page: number;
  similarity: number;
}

const API_BASE = 'http://127.0.0.1:8000';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  // Send message
  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input })
      });

      const data = await response.json();

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        confidence: data.confidence,
        citations: data.citations,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please make sure documents are uploaded and try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Ask example question
  const askExample = (question: string) => {
    setInput(question);
    setTimeout(() => sendMessage(), 100);
  };

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };

  // Upload files
  const uploadFiles = async () => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));

    try {
      const response = await fetch(`${API_BASE}/api/documents/upload`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      alert(`Successfully uploaded ${data.uploaded} document(s)!`);
      setShowUploadModal(false);
      setSelectedFiles([]);
    } catch (error) {
      alert('Upload failed: ' + error);
    } finally {
      setIsUploading(false);
    }
  };

  // Get confidence color
  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.7) return 'bg-green-100 text-green-800';
    if (confidence > 0.4) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <div className="flex min-h-screen flex-col bg-gray-50">
      {/* Header */}
      <header className="border-b bg-white shadow-sm">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <h1 className="text-xl font-semibold text-gray-900">
            ⚖️ Legal Document Q&A Assistant
          </h1>
          <button
            onClick={() => setShowUploadModal(true)}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            Upload Documents
          </button>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex flex-1 flex-col mx-auto w-full max-w-5xl px-6">
        {/* Messages */}
        <div
          ref={chatAreaRef}
          className="flex-1 overflow-y-auto py-8 space-y-6"
        >
          {messages.length === 0 ? (
            // Welcome Screen
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                How can I help with your legal questions?
              </h2>
              <p className="text-gray-600 mb-8 max-w-md">
                Upload legal documents and ask questions to get instant answers with citations.
              </p>

              {/* Example Questions */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
                <button
                  onClick={() => askExample('What are the remedies for breach of contract?')}
                  className="p-4 text-left bg-white border border-gray-200 rounded-xl hover:border-blue-500 hover:shadow-md transition-all"
                >
                  <div className="font-medium text-gray-900 mb-1">📋 Contract Law</div>
                  <div className="text-sm text-gray-600">Remedies for breach of contract</div>
                </button>

                <button
                  onClick={() => askExample('What is the statute of limitations for filing a lawsuit?')}
                  className="p-4 text-left bg-white border border-gray-200 rounded-xl hover:border-blue-500 hover:shadow-md transition-all"
                >
                  <div className="font-medium text-gray-900 mb-1">⏰ Legal Procedures</div>
                  <div className="text-sm text-gray-600">Statute of limitations</div>
                </button>

                <button
                  onClick={() => askExample('What are the key elements of negligence?')}
                  className="p-4 text-left bg-white border border-gray-200 rounded-xl hover:border-blue-500 hover:shadow-md transition-all"
                >
                  <div className="font-medium text-gray-900 mb-1">⚖️ Tort Law</div>
                  <div className="text-sm text-gray-600">Elements of negligence</div>
                </button>

                <button
                  onClick={() => askExample('What constitutes a valid contract?')}
                  className="p-4 text-left bg-white border border-gray-200 rounded-xl hover:border-blue-500 hover:shadow-md transition-all"
                >
                  <div className="font-medium text-gray-900 mb-1">📝 Contract Formation</div>
                  <div className="text-sm text-gray-600">Valid contract requirements</div>
                </button>
              </div>
            </div>
          ) : (
            // Messages
            messages.map((message, index) => (
              <div key={index} className="flex gap-4 animate-fadeIn">
                {/* Avatar */}
                <div className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-sm font-semibold ${
                  message.role === 'user' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'bg-purple-100 text-purple-700'
                }`}>
                  {message.role === 'user' ? 'U' : 'AI'}
                </div>

                {/* Message Content */}
                <div className="flex-1">
                  <div className="prose max-w-none">
                    <p className="text-gray-900 whitespace-pre-wrap">{message.content}</p>
                  </div>

                  {/* Confidence Badge */}
                  {message.confidence !== undefined && (
                    <div className={`inline-block mt-2 px-3 py-1 rounded-full text-xs font-medium ${getConfidenceColor(message.confidence)}`}>
                      Confidence: {Math.round(message.confidence * 100)}%
                    </div>
                  )}

                  {/* Citations */}
                  {message.citations && message.citations.length > 0 && (
                    <div className="mt-4 p-4 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg">
                      <div className="text-sm font-semibold text-gray-900 mb-2">📚 Sources:</div>
                      <div className="space-y-1">
                        {message.citations.map((citation, idx) => (
                          <div key={idx} className="text-sm text-gray-700">
                            • {citation.fileName} (Page {citation.page}) - {Math.round(citation.similarity * 100)}% relevant
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-4 animate-fadeIn">
              <div className="flex-shrink-0 w-9 h-9 rounded-full bg-purple-100 text-purple-700 flex items-center justify-center text-sm font-semibold">
                AI
              </div>
              <div className="flex-1">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t bg-white py-4 sticky bottom-0">
          <div className="flex gap-3 items-end">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask a question about your documents..."
              className="flex-1 resize-none rounded-lg border border-gray-300 px-4 py-3 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 max-h-40"
              rows={1}
              style={{ minHeight: '48px' }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = target.scrollHeight + 'px';
              }}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="flex-shrink-0 rounded-lg bg-blue-600 p-3 text-white hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </div>
      </main>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl max-w-md w-full p-8 shadow-2xl">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Legal Documents</h2>
            <p className="text-gray-600 mb-6">Upload PDF documents to ask questions about their content.</p>

            {/* File Upload Area */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50/50 transition-all"
            >
              <div className="text-5xl mb-4">📄</div>
              <div className="text-gray-700">Click to browse or drag and drop PDF files here</div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />

            {/* Selected Files */}
            {selectedFiles.length > 0 && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <div className="font-medium text-gray-900 mb-2">Selected files:</div>
                <div className="space-y-1">
                  {selectedFiles.map((file, idx) => (
                    <div key={idx} className="text-sm text-gray-600">• {file.name}</div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  setShowUploadModal(false);
                  setSelectedFiles([]);
                }}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg font-medium text-gray-700 hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={uploadFiles}
                disabled={selectedFiles.length === 0 || isUploading}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {isUploading ? 'Uploading...' : 'Upload'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}