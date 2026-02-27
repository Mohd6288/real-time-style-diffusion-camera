/* ═══════════════════════════════════════
   AI STYLE PRESETS
   Matching the original Stable Diffusion
   img2img prompts from the Python app.
═══════════════════════════════════════ */

export interface AIStylePreset {
  id: string;
  name: string;
  prompt: string;
  strength: number;
  guidanceScale: number;
  steps: number;
}

export const AI_STYLE_PRESETS: Record<string, AIStylePreset> = {
  vintage: {
    id: "vintage",
    name: "Old Vintage Photo",
    prompt:
      "a realistic old vintage photograph, sepia tone, faded colors, film grain, " +
      "scratches, dust, soft blur, aged paper texture, 1950s analog camera aesthetic, retro historical photo",
    strength: 0.6,
    guidanceScale: 7.5,
    steps: 20,
  },
  bw: {
    id: "bw",
    name: "1920s Black & White",
    prompt:
      "1920s black and white portrait, high contrast, soft lens blur, film grain, " +
      "old cinema look, silver gelatin print texture, historical vintage photograph",
    strength: 0.6,
    guidanceScale: 7.5,
    steps: 20,
  },
  retro: {
    id: "retro",
    name: "1970s Retro Film",
    prompt:
      "retro 1970s photograph, faded warm colors, light leaks, soft blur, vintage film grain, " +
      "analog lens distortion, nostalgic kodak aesthetic",
    strength: 0.6,
    guidanceScale: 7.5,
    steps: 20,
  },
  comic: {
    id: "comic",
    name: "Comic Book Style",
    prompt:
      "comic book hero style, dramatic ink outlines, halftone shading, bold colors, dynamic lighting, " +
      "sharp shadows, retro comic texture, graphic novel illustration",
    strength: 0.65,
    guidanceScale: 8.0,
    steps: 25,
  },
  cartoon: {
    id: "cartoon",
    name: "2D Cartoon",
    prompt:
      "flat 2D cartoon character style, minimal shading, bold outlines, thick black line art, " +
      "simple shapes, vibrant solid colors, vector-style illustration, modern cartoon aesthetic",
    strength: 0.65,
    guidanceScale: 8.0,
    steps: 25,
  },
  anime: {
    id: "anime",
    name: "Anime Style",
    prompt:
      "anime-style portrait, big glossy eyes, detailed hair, clean cel shading, vivid colors, soft glow, " +
      "studio anime illustration, perfect linework",
    strength: 0.65,
    guidanceScale: 8.0,
    steps: 25,
  },
};
