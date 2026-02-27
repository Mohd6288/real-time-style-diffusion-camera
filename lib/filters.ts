/* ═══════════════════════════════════════
   CANVAS STYLE FILTERS
   Each filter operates on raw ImageData
   pixel arrays for maximum quality.
═══════════════════════════════════════ */

export interface StylePreset {
  id: string;
  name: string;
  description: string;
  cssFilter: string;        // live preview on <video>
  emoji: string;
  color: string;
  apply: (ctx: CanvasRenderingContext2D, w: number, h: number) => void;
}

// ── Pixel helpers ──────────────────────

function getPixels(ctx: CanvasRenderingContext2D, w: number, h: number) {
  return ctx.getImageData(0, 0, w, h);
}

function putPixels(ctx: CanvasRenderingContext2D, img: ImageData) {
  ctx.putImageData(img, 0, 0);
}

function clamp(v: number) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// ── Vignette overlay ──────────────────

function applyVignette(ctx: CanvasRenderingContext2D, w: number, h: number, strength = 0.5) {
  const cx = w / 2, cy = h / 2;
  const r = Math.max(cx, cy);
  const grad = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, r * 1.1);
  grad.addColorStop(0, "transparent");
  grad.addColorStop(1, `rgba(0,0,0,${strength})`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
}

// ── Noise / grain ─────────────────────

function applyGrain(ctx: CanvasRenderingContext2D, w: number, h: number, amount = 25, opacity = 0.3) {
  const img = getPixels(ctx, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (Math.random() - 0.5) * amount;
    d[i]     = clamp(d[i] + n);
    d[i + 1] = clamp(d[i + 1] + n);
    d[i + 2] = clamp(d[i + 2] + n);
  }
  // blend grain at reduced opacity
  ctx.globalAlpha = opacity;
  putPixels(ctx, img);
  ctx.globalAlpha = 1;
}

// ═══════════════════════════════════════
//  FILTER 1: VINTAGE FILM
// ═══════════════════════════════════════

function vintageFilm(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const img = getPixels(ctx, w, h);
  const d = img.data;

  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    // Sepia matrix
    d[i]     = clamp(r * 0.393 + g * 0.769 + b * 0.189);
    d[i + 1] = clamp(r * 0.349 + g * 0.686 + b * 0.168);
    d[i + 2] = clamp(r * 0.272 + g * 0.534 + b * 0.131);
    // Fade slightly
    d[i]     = clamp(d[i] * 0.9 + 25);
    d[i + 1] = clamp(d[i + 1] * 0.85 + 20);
    d[i + 2] = clamp(d[i + 2] * 0.8 + 15);
  }
  putPixels(ctx, img);
  applyVignette(ctx, w, h, 0.6);
  applyGrain(ctx, w, h, 35, 0.4);
}

// ═══════════════════════════════════════
//  FILTER 2: CLASSIC B&W
// ═══════════════════════════════════════

function classicBW(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const img = getPixels(ctx, w, h);
  const d = img.data;

  for (let i = 0; i < d.length; i += 4) {
    // Luminosity grayscale
    let gray = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
    // Increase contrast
    gray = clamp((gray - 128) * 1.4 + 128);
    d[i] = d[i + 1] = d[i + 2] = gray;
  }
  putPixels(ctx, img);
  applyVignette(ctx, w, h, 0.45);
  applyGrain(ctx, w, h, 30, 0.35);
}

// ═══════════════════════════════════════
//  FILTER 3: RETRO 70s
// ═══════════════════════════════════════

function retro70s(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const img = getPixels(ctx, w, h);
  const d = img.data;

  for (let i = 0; i < d.length; i += 4) {
    // Warm shift + faded blacks
    d[i]     = clamp(d[i] * 1.1 + 15);      // boost red
    d[i + 1] = clamp(d[i + 1] * 1.05 + 8);  // slight green
    d[i + 2] = clamp(d[i + 2] * 0.85);       // reduce blue
    // Fade black point (raise shadows)
    d[i]     = clamp(d[i] * 0.9 + 30);
    d[i + 1] = clamp(d[i + 1] * 0.9 + 25);
    d[i + 2] = clamp(d[i + 2] * 0.9 + 20);
  }
  putPixels(ctx, img);

  // Light leak — warm gradient from top-right
  const grad = ctx.createLinearGradient(w * 0.6, 0, w, h * 0.5);
  grad.addColorStop(0, "rgba(255,180,50,0.25)");
  grad.addColorStop(0.5, "rgba(255,100,50,0.1)");
  grad.addColorStop(1, "transparent");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  applyVignette(ctx, w, h, 0.35);
  applyGrain(ctx, w, h, 20, 0.25);
}

// ═══════════════════════════════════════
//  FILTER 4: COMIC BOOK
// ═══════════════════════════════════════

function comicBook(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const img = getPixels(ctx, w, h);
  const d = img.data;
  const levels = 5;
  const step = 255 / levels;

  // Pass 1: Posterize
  for (let i = 0; i < d.length; i += 4) {
    d[i]     = Math.round(d[i] / step) * step;
    d[i + 1] = Math.round(d[i + 1] / step) * step;
    d[i + 2] = Math.round(d[i + 2] / step) * step;
    // Boost saturation
    const gray = (d[i] + d[i + 1] + d[i + 2]) / 3;
    d[i]     = clamp(gray + (d[i] - gray) * 1.6);
    d[i + 1] = clamp(gray + (d[i + 1] - gray) * 1.6);
    d[i + 2] = clamp(gray + (d[i + 2] - gray) * 1.6);
  }
  putPixels(ctx, img);

  // Pass 2: Edge detection overlay
  const src = getPixels(ctx, w, h);
  const edge = ctx.createImageData(w, h);
  const sd = src.data, ed = edge.data;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      // Sobel-like (simplified)
      const l = (sd[idx - 4] + sd[idx - 3] + sd[idx - 2]) / 3;
      const r = (sd[idx + 4] + sd[idx + 5] + sd[idx + 6]) / 3;
      const t = (sd[((y - 1) * w + x) * 4] + sd[((y - 1) * w + x) * 4 + 1] + sd[((y - 1) * w + x) * 4 + 2]) / 3;
      const b = (sd[((y + 1) * w + x) * 4] + sd[((y + 1) * w + x) * 4 + 1] + sd[((y + 1) * w + x) * 4 + 2]) / 3;
      const gx = Math.abs(r - l);
      const gy = Math.abs(b - t);
      const mag = gx + gy;
      const ink = mag > 40 ? 0 : 255;
      ed[idx] = ed[idx + 1] = ed[idx + 2] = ink;
      ed[idx + 3] = 255;
    }
  }

  // Blend edges on top
  ctx.globalCompositeOperation = "multiply";
  putPixels(ctx, edge);
  ctx.globalCompositeOperation = "source-over";
}

// ═══════════════════════════════════════
//  FILTER 5: CARTOON
// ═══════════════════════════════════════

function cartoon(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const img = getPixels(ctx, w, h);
  const d = img.data;
  const levels = 8;
  const step = 255 / levels;

  // Quantize colors + boost saturation
  for (let i = 0; i < d.length; i += 4) {
    d[i]     = Math.round(d[i] / step) * step;
    d[i + 1] = Math.round(d[i + 1] / step) * step;
    d[i + 2] = Math.round(d[i + 2] / step) * step;
    // Saturate
    const gray = (d[i] + d[i + 1] + d[i + 2]) / 3;
    d[i]     = clamp(gray + (d[i] - gray) * 1.8);
    d[i + 1] = clamp(gray + (d[i + 1] - gray) * 1.8);
    d[i + 2] = clamp(gray + (d[i + 2] - gray) * 1.8);
    // Brighten slightly
    d[i]     = clamp(d[i] + 10);
    d[i + 1] = clamp(d[i + 1] + 10);
    d[i + 2] = clamp(d[i + 2] + 10);
  }
  putPixels(ctx, img);

  // Soft edge overlay
  const src = getPixels(ctx, w, h);
  const edge = ctx.createImageData(w, h);
  const sd = src.data, ed = edge.data;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      const l = (sd[idx - 4] + sd[idx - 3] + sd[idx - 2]) / 3;
      const r = (sd[idx + 4] + sd[idx + 5] + sd[idx + 6]) / 3;
      const t = (sd[((y - 1) * w + x) * 4] + sd[((y - 1) * w + x) * 4 + 1] + sd[((y - 1) * w + x) * 4 + 2]) / 3;
      const b = (sd[((y + 1) * w + x) * 4] + sd[((y + 1) * w + x) * 4 + 1] + sd[((y + 1) * w + x) * 4 + 2]) / 3;
      const mag = Math.abs(r - l) + Math.abs(b - t);
      const ink = mag > 30 ? 60 : 255;
      ed[idx] = ed[idx + 1] = ed[idx + 2] = ink;
      ed[idx + 3] = 255;
    }
  }

  ctx.globalCompositeOperation = "multiply";
  putPixels(ctx, edge);
  ctx.globalCompositeOperation = "source-over";
}

// ═══════════════════════════════════════
//  FILTER 6: ANIME
// ═══════════════════════════════════════

function anime(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const img = getPixels(ctx, w, h);
  const d = img.data;

  for (let i = 0; i < d.length; i += 4) {
    // Boost saturation heavily
    const gray = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
    d[i]     = clamp(gray + (d[i] - gray) * 2.0);
    d[i + 1] = clamp(gray + (d[i + 1] - gray) * 2.0);
    d[i + 2] = clamp(gray + (d[i + 2] - gray) * 2.0);
    // Enhance contrast
    d[i]     = clamp((d[i] - 128) * 1.25 + 128);
    d[i + 1] = clamp((d[i + 1] - 128) * 1.25 + 128);
    d[i + 2] = clamp((d[i + 2] - 128) * 1.25 + 128);
    // Purple-tint shadows
    if (gray < 80) {
      d[i]     = clamp(d[i] + 8);
      d[i + 2] = clamp(d[i + 2] + 18);
    }
    // Slight quantize for cel-shading look
    const q = 32;
    d[i]     = Math.round(d[i] / q) * q;
    d[i + 1] = Math.round(d[i + 1] / q) * q;
    d[i + 2] = Math.round(d[i + 2] / q) * q;
  }
  putPixels(ctx, img);

  // Subtle edge overlay
  const src = getPixels(ctx, w, h);
  const edge = ctx.createImageData(w, h);
  const sd = src.data, ed = edge.data;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      const l = (sd[idx - 4] + sd[idx - 3] + sd[idx - 2]) / 3;
      const r = (sd[idx + 4] + sd[idx + 5] + sd[idx + 6]) / 3;
      const t = (sd[((y - 1) * w + x) * 4] + sd[((y - 1) * w + x) * 4 + 1] + sd[((y - 1) * w + x) * 4 + 2]) / 3;
      const b = (sd[((y + 1) * w + x) * 4] + sd[((y + 1) * w + x) * 4 + 1] + sd[((y + 1) * w + x) * 4 + 2]) / 3;
      const mag = Math.abs(r - l) + Math.abs(b - t);
      const ink = mag > 35 ? 40 : 255;
      ed[idx] = ed[idx + 1] = ed[idx + 2] = ink;
      ed[idx + 3] = 255;
    }
  }

  ctx.globalCompositeOperation = "multiply";
  putPixels(ctx, edge);
  ctx.globalCompositeOperation = "source-over";
}

// ═══════════════════════════════════════
//  EXPORTS
// ═══════════════════════════════════════

export const PRESETS: StylePreset[] = [
  {
    id: "vintage",
    name: "Vintage Film",
    description: "Sepia tone, faded colors, film grain, and vignette — 1950s aesthetic",
    cssFilter: "sepia(0.7) contrast(1.1) brightness(0.9) saturate(0.7)",
    emoji: "\uD83C\uDFDE\uFE0F",
    color: "#d4a574",
    apply: vintageFilm,
  },
  {
    id: "bw",
    name: "Classic B&W",
    description: "High-contrast grayscale with silver gelatin grain texture",
    cssFilter: "grayscale(1) contrast(1.35) brightness(1.05)",
    emoji: "\uD83D\uDDA4",
    color: "#9ca3af",
    apply: classicBW,
  },
  {
    id: "retro",
    name: "Retro 70s",
    description: "Warm faded colors, light leaks, and analog Kodak aesthetic",
    cssFilter: "sepia(0.25) saturate(1.4) brightness(1.1) hue-rotate(-10deg)",
    emoji: "\u2728",
    color: "#f59e0b",
    apply: retro70s,
  },
  {
    id: "comic",
    name: "Comic Book",
    description: "Bold ink outlines, posterized colors, and halftone shading",
    cssFilter: "contrast(1.8) saturate(1.6) brightness(1.05)",
    emoji: "\uD83D\uDCA5",
    color: "#ef4444",
    apply: comicBook,
  },
  {
    id: "cartoon",
    name: "Cartoon",
    description: "Flat bold colors, thick outlines, and vibrant saturation",
    cssFilter: "contrast(1.4) saturate(1.7) brightness(1.08)",
    emoji: "\uD83C\uDFA8",
    color: "#22c55e",
    apply: cartoon,
  },
  {
    id: "anime",
    name: "Anime",
    description: "Cel-shaded look with vivid colors, edge lines, and purple-tinted shadows",
    cssFilter: "saturate(1.9) contrast(1.2) brightness(1.05) hue-rotate(5deg)",
    emoji: "\u2B50",
    color: "#a855f7",
    apply: anime,
  },
];
