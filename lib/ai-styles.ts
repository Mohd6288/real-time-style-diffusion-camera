/* ═══════════════════════════════════════
   AI STYLE PRESETS
   Prompts for FLUX.1-Kontext-dev
   image-to-image style transfer.
═══════════════════════════════════════ */

export interface AIStylePreset {
  id: string;
  name: string;
  prompt: string;
  guidanceScale: number;
  steps: number;
}

export const AI_STYLE_PRESETS: Record<string, AIStylePreset> = {
  vintage: {
    id: "vintage",
    name: "Old Vintage Photo",
    prompt:
      "Transform this image into a realistic old vintage photograph with sepia tone, faded colors, film grain, " +
      "scratches, dust, soft blur, aged paper texture, 1950s analog camera aesthetic, retro historical photo",
    guidanceScale: 7.5,
    steps: 28,
  },
  bw: {
    id: "bw",
    name: "1920s Black & White",
    prompt:
      "Transform this image into a 1920s black and white photograph with high contrast, soft lens blur, film grain, " +
      "old cinema look, silver gelatin print texture, historical vintage photograph style",
    guidanceScale: 7.5,
    steps: 28,
  },
  retro: {
    id: "retro",
    name: "1970s Retro Film",
    prompt:
      "Transform this image into a retro 1970s photograph with faded warm colors, light leaks, soft blur, vintage film grain, " +
      "analog lens distortion, nostalgic Kodak aesthetic",
    guidanceScale: 7.5,
    steps: 28,
  },
  comic: {
    id: "comic",
    name: "Comic Book Style",
    prompt:
      "Transform this image into comic book art style with dramatic ink outlines, halftone shading, bold colors, dynamic lighting, " +
      "sharp shadows, retro comic texture, graphic novel illustration",
    guidanceScale: 8.0,
    steps: 28,
  },
  cartoon: {
    id: "cartoon",
    name: "2D Cartoon",
    prompt:
      "Transform this image into a flat 2D cartoon character style with minimal shading, bold outlines, thick black line art, " +
      "simple shapes, vibrant solid colors, vector-style illustration, modern cartoon aesthetic",
    guidanceScale: 8.0,
    steps: 28,
  },
  anime: {
    id: "anime",
    name: "Anime Style",
    prompt:
      "Transform this image into anime-style art with big glossy eyes, detailed hair, clean cel shading, vivid colors, soft glow, " +
      "studio anime illustration, perfect linework",
    guidanceScale: 8.0,
    steps: 28,
  },
};
