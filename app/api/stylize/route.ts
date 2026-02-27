import { NextRequest, NextResponse } from "next/server";
import { InferenceClient } from "@huggingface/inference";
import { AI_STYLE_PRESETS } from "@/lib/ai-styles";

const hf = new InferenceClient(process.env.HF_TOKEN!);

export async function POST(req: NextRequest) {
  try {
    // Validate token
    if (!process.env.HF_TOKEN) {
      return NextResponse.json(
        { error: "HuggingFace API token not configured. Set HF_TOKEN in environment variables." },
        { status: 500 }
      );
    }

    const body = await req.json();
    const { image, styleId, extraPrompt } = body as {
      image: string;     // base64 data URL
      styleId: string;
      extraPrompt?: string;
    };

    if (!image || !styleId) {
      return NextResponse.json(
        { error: "Missing image or styleId" },
        { status: 400 }
      );
    }

    const preset = AI_STYLE_PRESETS[styleId];
    if (!preset) {
      return NextResponse.json(
        { error: `Unknown style: ${styleId}` },
        { status: 400 }
      );
    }

    // Build prompt
    let prompt = preset.prompt;
    if (extraPrompt?.trim()) {
      prompt = `${prompt}, ${extraPrompt.trim()}`;
    }

    // Convert base64 data URL to Blob
    const base64Data = image.replace(/^data:image\/\w+;base64,/, "");
    const imageBuffer = Buffer.from(base64Data, "base64");
    const imageBlob = new Blob([imageBuffer], { type: "image/png" });

    // Call HuggingFace FLUX.1-Kontext-dev img2img via Replicate provider
    const result = await hf.imageToImage({
      model: "black-forest-labs/FLUX.1-Kontext-dev",
      provider: "replicate",
      inputs: imageBlob,
      parameters: {
        prompt,
        guidance_scale: preset.guidanceScale,
        num_inference_steps: preset.steps,
      },
    });

    // Convert result Blob to base64
    const arrayBuffer = await result.arrayBuffer();
    const resultBase64 = Buffer.from(arrayBuffer).toString("base64");
    const resultDataUrl = `data:image/png;base64,${resultBase64}`;

    return NextResponse.json({ image: resultDataUrl });
  } catch (err: unknown) {
    console.error("Stylize API error:", err);
    const message = err instanceof Error ? err.message : "AI generation failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
