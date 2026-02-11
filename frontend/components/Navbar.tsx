'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { name: "Upload", href: "/upload" },
  { name: "Chat", href: "/chat" },
  { name: "Metrics", href: "/metrics" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="border-b bg-white">
      <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
        <div className="text-lg font-semibold">
          Legal Document QnA
        </div>

        <div className="flex gap-6">
          {navItems.map((item) => {
            const active = pathname === item.href;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`text-sm font-medium ${
                  active
                    ? "text-blue-600"
                    : "text-gray-600 hover:text-gray-900"
                }`}
              >
                {item.name}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
