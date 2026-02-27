import Link from "next/link";

const STYLES = [
  { name: "Vintage Film",  desc: "AI transforms your photo into a sepia-toned 1950s analog photograph with grain and scratches",    emoji: "\uD83C\uDFDE\uFE0F", color: "#d4a574" },
  { name: "Classic B&W",   desc: "AI recreates the look of a 1920s silver gelatin print with high contrast and film noir feel",  emoji: "\uD83D\uDDA4", color: "#9ca3af" },
  { name: "Retro 70s",     desc: "AI applies warm faded Kodak colors with light leaks and analog lens distortion",               emoji: "\u2728", color: "#f59e0b" },
  { name: "Comic Book",    desc: "AI generates dramatic ink outlines, halftone shading, and bold pop-art colors",           emoji: "\uD83D\uDCA5", color: "#ef4444" },
  { name: "Cartoon",       desc: "AI creates flat 2D cartoon style with bold outlines and vibrant solid colors",                  emoji: "\uD83C\uDFA8", color: "#22c55e" },
  { name: "Anime",         desc: "AI produces cel-shaded anime art with vivid colors, clean linework, and soft glow",     emoji: "\u2B50", color: "#a855f7" },
];

const FEATURES = [
  { title: "AI Style Transfer",    desc: "Stable Diffusion v1.5 transforms your photos into genuine AI-generated art.", icon: "\uD83E\uDD16" },
  { title: "6 Artistic Styles",    desc: "From vintage film to anime — each powered by carefully crafted AI prompts.", icon: "\uD83C\uDFA8" },
  { title: "Instant Preview",      desc: "Real-time CSS & canvas filters let you preview styles at 60fps before generating.", icon: "\u26A1" },
  { title: "High-Quality Output",  desc: "Download AI-generated or canvas-filtered results as high-quality PNG.", icon: "\uD83D\uDCF7" },
];

export default function Home() {
  return (
    <div className="fade-up">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-[var(--border)]">
        <div className="absolute inset-0 opacity-[0.04]" style={{ backgroundImage: "radial-gradient(circle, var(--accent) 1px, transparent 1px)", backgroundSize: "30px 30px" }} />
        <div className="max-w-5xl mx-auto px-5 py-24 text-center relative z-10">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[var(--surface)] border border-[var(--border)] text-[var(--accent)] text-xs font-semibold mb-6">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Powered by Stable Diffusion &middot; AI Style Transfer &middot; Real-Time Preview
          </div>
          <h1 className="text-4xl md:text-6xl font-black text-[var(--fg)] mb-4 leading-tight">
            AI Style Diffusion<br />
            <span className="text-[var(--accent)]">Camera</span>
          </h1>
          <p className="text-[var(--muted)] text-base md:text-lg max-w-xl mx-auto mb-10 leading-relaxed">
            Transform your webcam feed or photos into stunning AI-generated art. Stable Diffusion turns your images into vintage film, comic book, anime, and more.
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <Link href="/studio" className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl bg-[var(--accent)] text-white text-sm font-bold shadow-lg hover:bg-[var(--accent2)] hover:scale-[1.03] transition-all" style={{ animation: "pulse-glow 3s ease-in-out infinite" }}>
              &#127912; Open AI Studio
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-5xl mx-auto px-5 py-16">
        <h2 className="text-2xl font-bold text-center text-[var(--fg)] mb-10">How It Works</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map((f) => (
            <div key={f.title} className="p-5 rounded-2xl bg-[var(--surface)] border border-[var(--border)] hover:border-[var(--accent)]/40 transition-colors">
              <div className="text-2xl mb-3">{f.icon}</div>
              <h3 className="text-sm font-bold text-[var(--fg)] mb-1">{f.title}</h3>
              <p className="text-xs text-[var(--muted)] leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Style Grid */}
      <section className="border-t border-[var(--border)] bg-[var(--surface)]/50">
        <div className="max-w-5xl mx-auto px-5 py-16">
          <h2 className="text-2xl font-bold text-center text-[var(--fg)] mb-3">6 AI Art Styles</h2>
          <p className="text-sm text-center text-[var(--muted)] mb-10 max-w-md mx-auto">
            Each style uses Stable Diffusion img2img with carefully tuned prompts to transform your photos into genuine AI-generated artwork.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {STYLES.map((s) => (
              <div key={s.name} className="p-5 rounded-2xl bg-[var(--card)] border border-[var(--border)] hover:border-[var(--accent)]/40 hover:-translate-y-1 transition-all">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-10 h-10 rounded-xl flex items-center justify-center text-lg" style={{ background: s.color + "20" }}>
                    {s.emoji}
                  </span>
                  <h3 className="text-sm font-bold" style={{ color: s.color }}>{s.name}</h3>
                </div>
                <p className="text-xs text-[var(--muted)] leading-relaxed">{s.desc}</p>
              </div>
            ))}
          </div>
          <div className="text-center mt-10">
            <Link href="/studio" className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-[var(--accent)] text-white text-sm font-bold hover:bg-[var(--accent2)] transition-all">
              Try AI Studio Now &rarr;
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
