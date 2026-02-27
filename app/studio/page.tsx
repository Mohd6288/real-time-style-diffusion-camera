"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { PRESETS, type StylePreset } from "@/lib/filters";

type Mode = "idle" | "camera" | "upload";

export default function StudioPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resultRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [mode, setMode] = useState<Mode>("idle");
  const [selected, setSelected] = useState<StylePreset>(PRESETS[0]);
  const [captured, setCaptured] = useState(false);
  const [error, setError] = useState("");

  // ── Start camera ──
  const startCamera = useCallback(async () => {
    setError("");
    setCaptured(false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setMode("camera");
    } catch {
      setError("Camera access denied. Please allow camera permission and try again.");
    }
  }, []);

  // ── Stop camera ──
  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setMode("idle");
    setCaptured(false);
  }, []);

  // ── Handle file upload ──
  const handleUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError("");
    setCaptured(false);

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      // Fit to max 640x480
      const scale = Math.min(640 / img.width, 480 / img.height, 1);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      setMode("upload");
    };
    img.src = URL.createObjectURL(file);
  }, []);

  // ── Capture & apply filter ──
  const capture = useCallback(() => {
    const result = resultRef.current;
    if (!result) return;
    const ctx = result.getContext("2d")!;

    if (mode === "camera" && videoRef.current) {
      const v = videoRef.current;
      result.width = v.videoWidth;
      result.height = v.videoHeight;
      ctx.drawImage(v, 0, 0);
    } else if (mode === "upload" && canvasRef.current) {
      const src = canvasRef.current;
      result.width = src.width;
      result.height = src.height;
      ctx.drawImage(src, 0, 0);
    }

    // Apply the selected canvas filter
    selected.apply(ctx, result.width, result.height);
    setCaptured(true);
  }, [mode, selected]);

  // ── Download result ──
  const download = useCallback(() => {
    const canvas = resultRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `styled-${selected.id}-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }, [selected]);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div className="fade-up max-w-6xl mx-auto px-5 py-8">
      {/* Title */}
      <div className="text-center mb-8">
        <h1 className="text-2xl md:text-3xl font-black text-[var(--fg)] mb-2">
          &#127912; Style Studio
        </h1>
        <p className="text-sm text-[var(--muted)]">
          Capture from camera or upload an image, pick a style, and transform.
        </p>
      </div>

      {error && (
        <div className="mb-6 p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-sm text-red-400 text-center">
          {error}
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-6">
        {/* ── Left: Controls ── */}
        <div className="space-y-4">
          {/* Source buttons */}
          <div className="p-4 rounded-2xl bg-[var(--surface)] border border-[var(--border)]">
            <h3 className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider mb-3">Source</h3>
            <div className="flex gap-2">
              {mode === "camera" ? (
                <button onClick={stopCamera} className="flex-1 py-2.5 rounded-xl text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-all">
                  Stop Camera
                </button>
              ) : (
                <button onClick={startCamera} className="flex-1 py-2.5 rounded-xl text-xs font-bold bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/30 hover:bg-[var(--accent)]/30 transition-all">
                  &#128247; Camera
                </button>
              )}
              <button
                onClick={() => fileRef.current?.click()}
                className="flex-1 py-2.5 rounded-xl text-xs font-bold bg-[var(--card)] text-[var(--muted)] border border-[var(--border)] hover:border-[var(--accent)]/40 hover:text-[var(--fg)] transition-all"
              >
                &#128193; Upload
              </button>
              <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleUpload} />
            </div>
          </div>

          {/* Style presets */}
          <div className="p-4 rounded-2xl bg-[var(--surface)] border border-[var(--border)]">
            <h3 className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider mb-3">Style</h3>
            <div className="grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => { setSelected(p); setCaptured(false); }}
                  className={`p-3 rounded-xl text-left transition-all ${
                    selected.id === p.id
                      ? "bg-[var(--accent)]/20 border-2 border-[var(--accent)] shadow-[0_0_12px_rgba(139,92,246,0.2)]"
                      : "bg-[var(--card)] border border-[var(--border)] hover:border-[var(--accent)]/40"
                  }`}
                >
                  <span className="text-lg">{p.emoji}</span>
                  <p className="text-xs font-bold mt-1" style={{ color: selected.id === p.id ? p.color : "var(--fg)" }}>
                    {p.name}
                  </p>
                </button>
              ))}
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={capture}
              disabled={mode === "idle"}
              className="flex-1 py-3 rounded-xl text-sm font-bold bg-[var(--accent)] text-white hover:bg-[var(--accent2)] disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              &#128248; Capture &amp; Style
            </button>
            {captured && (
              <button
                onClick={download}
                className="py-3 px-5 rounded-xl text-sm font-bold bg-green-600 text-white hover:bg-green-500 transition-all"
              >
                &#11015;&#65039;
              </button>
            )}
          </div>
        </div>

        {/* ── Center: Live preview ── */}
        <div className="lg:col-span-2 space-y-4">
          {/* Live video / upload preview */}
          <div className="rounded-2xl bg-[var(--surface)] border border-[var(--border)] overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2 border-b border-[var(--border)]">
              <span className={`w-2 h-2 rounded-full ${mode !== "idle" ? "bg-green-500 animate-pulse" : "bg-[var(--border)]"}`} />
              <span className="text-xs font-semibold text-[var(--muted)]">
                {mode === "camera" ? "Live Preview" : mode === "upload" ? "Uploaded Image" : "No Source"}
              </span>
              <span className="text-xs text-[var(--muted)] ml-auto">{selected.name}</span>
            </div>
            <div className="relative bg-black flex items-center justify-center" style={{ minHeight: 300 }}>
              {/* Camera video (with CSS filter for live preview) */}
              <video
                ref={videoRef}
                playsInline
                muted
                className="max-w-full max-h-[480px]"
                style={{
                  display: mode === "camera" ? "block" : "none",
                  filter: selected.cssFilter,
                  transition: "filter 0.3s ease",
                }}
              />

              {/* Uploaded image canvas (with CSS filter for preview) */}
              <canvas
                ref={canvasRef}
                className="max-w-full max-h-[480px]"
                style={{
                  display: mode === "upload" ? "block" : "none",
                  filter: selected.cssFilter,
                  transition: "filter 0.3s ease",
                }}
              />

              {/* Idle state */}
              {mode === "idle" && (
                <div className="text-center py-20">
                  <p className="text-4xl mb-3">&#127912;</p>
                  <p className="text-sm text-[var(--muted)]">Start camera or upload an image to begin</p>
                </div>
              )}
            </div>
          </div>

          {/* Captured result */}
          {captured && (
            <div className="rounded-2xl bg-[var(--surface)] border border-[var(--border)] overflow-hidden">
              <div className="flex items-center gap-2 px-4 py-2 border-b border-[var(--border)]">
                <span className="w-2 h-2 rounded-full bg-[var(--accent)]" />
                <span className="text-xs font-semibold text-[var(--muted)]">Styled Result</span>
                <span className="text-xs font-semibold ml-auto" style={{ color: selected.color }}>{selected.name}</span>
              </div>
              <div className="bg-black flex items-center justify-center">
                <canvas
                  ref={resultRef}
                  className="max-w-full max-h-[480px]"
                />
              </div>
              <div className="px-4 py-3 border-t border-[var(--border)] flex items-center justify-between">
                <p className="text-xs text-[var(--muted)]">
                  Canvas-processed with edge detection, color quantization &amp; grain
                </p>
                <button
                  onClick={download}
                  className="text-xs font-bold text-[var(--accent)] hover:text-[var(--accent2)] transition-colors"
                >
                  Download PNG
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
