import "./globals.css";
import type { Metadata } from "next";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "Legal Document QnA System",
  description: "RAG based legal document analysis system",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50">
        <Navbar />
        <main className="mx-auto max-w-7xl px-6 py-6">
          {children}
        </main>
      </body>
    </html>
  );
}
