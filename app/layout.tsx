import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Style Diffusion Camera",
  description: "Real-time artistic style transfer — transform your camera feed into vintage film, comic book, anime, and more.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        {/* Header */}
        <header className="sticky top-0 z-50 backdrop-blur-xl bg-[var(--bg)]/90 border-b border-[var(--border)]">
          <nav className="max-w-6xl mx-auto flex items-center justify-between px-5 py-3">
            <Link href="/" className="flex items-center gap-2.5 group">
              <span className="w-8 h-8 rounded-lg bg-[var(--accent)] flex items-center justify-center text-white text-sm">
                &#127912;
              </span>
              <span className="font-bold text-sm text-[var(--fg)] group-hover:text-[var(--accent)] transition-colors hidden sm:block">
                Style Diffusion Camera
              </span>
            </Link>
            <div className="flex items-center gap-1">
              <Link href="/" className="px-3 py-2 rounded-lg text-sm font-medium text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--surface)] transition-all">
                Home
              </Link>
              <Link href="/studio" className="px-4 py-2 rounded-lg text-sm font-semibold bg-[var(--accent)] text-white hover:bg-[var(--accent2)] transition-all">
                Open Studio
              </Link>
            </div>
          </nav>
        </header>

        <main className="flex-1">{children}</main>

        <footer className="border-t border-[var(--border)]">
          <div className="max-w-6xl mx-auto px-5 py-6 flex flex-col sm:flex-row items-center justify-between gap-3">
            <span className="text-xs text-[var(--muted)]">Style Diffusion Camera &copy; {new Date().getFullYear()}</span>
            <span className="text-xs text-[var(--muted)]">Built with Next.js &middot; Runs entirely in your browser</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
