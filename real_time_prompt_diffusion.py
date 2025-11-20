import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np
import gradio as gr

# --------------------------
# 1. Model config
# --------------------------
MODEL_NAME = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model {MODEL_NAME} on {device} with dtype={dtype} ...")

# NOTE: If your diffusers version complains about `dtype`, change it to `torch_dtype`
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_NAME,
    dtype=dtype,          # use torch_dtype=dtype if you get a warning or error
)
pipe = pipe.to(device)

use_autocast = device == "cuda"

# --------------------------
# 2. Style presets
# --------------------------
STYLE_PRESETS = {
    "Old Vintage Photo": (
        "a realistic old vintage photograph, sepia tone, faded colors, film grain, "
        "scratches, dust, soft blur, aged paper texture, 1950s analog camera aesthetic, retro historical photo"
    ),
    "1920s Black & White": (
        "1920s black and white portrait, high contrast, soft lens blur, film grain, "
        "old cinema look, silver gelatin print texture, historical vintage photograph"
    ),
    "1970s Retro Film": (
        "retro 1970s photograph, faded warm colors, light leaks, soft blur, vintage film grain, "
        "analog lens distortion, nostalgic kodak aesthetic"
    ),
    "Comic Book Style": (
        "comic book hero style, dramatic ink outlines, halftone shading, bold colors, dynamic lighting, "
        "sharp shadows, retro comic texture, graphic novel illustration"
    ),
    "2D Cartoon": (
        "flat 2D cartoon character style, minimal shading, bold outlines, thick black line art, "
        "simple shapes, vibrant solid colors, vector-style illustration, modern cartoon aesthetic"
    ),
    "Anime Style": (
        "anime-style portrait, big glossy eyes, detailed hair, clean cel shading, vivid colors, soft glow, "
        "studio anime illustration, perfect linework"
    ),
}


# --------------------------
# 3. Core generation function
# --------------------------
def generate_from_cam(
    style_name: str,
    extra_prompt: str,
    init_image,
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
):
    """
    Apply a chosen style to the input image using Stable Diffusion img2img.
    - style_name: which preset to use from STYLE_PRESETS
    - extra_prompt: user text to refine / add details (optional)
    - init_image: webcam or uploaded image
    """
    if init_image is None:
        return None

    # Normalize to PIL.Image
    if isinstance(init_image, np.ndarray):
        init_image = Image.fromarray(init_image)

    init_image = init_image.convert("RGB")
    init_image = init_image.resize((448, 448))  # a bit smaller than 512 for speed

    base_prompt = STYLE_PRESETS.get(style_name, "")
    extra_prompt = (extra_prompt or "").strip()

    if extra_prompt:
        prompt = f"{base_prompt}, {extra_prompt}"
    else:
        prompt = base_prompt

    if use_autocast:
        with torch.autocast(device_type="cuda"):
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
    else:
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

    out_img = result.images[0]
    return out_img


# --------------------------
# 4. Gradio UI (dark mode)
# --------------------------
def build_interface():
    # Dark-ish theme
    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="indigo",
        neutral_hue="slate",
    ).set(
        body_background_fill="#020617",   # slate-950
        body_text_color="#e5e7eb",       # slate-200
        border_color_primary="#1f2937",  # slate-800
    )

    with gr.Blocks(
        title="Real-Time Style Diffusion Camera",
        theme=theme,
    ) as demo:
        gr.Markdown(
            """
            # 🎥 Real-Time Style Diffusion Camera

            Turn your webcam or uploaded images into **old photos, comics, cartoons, or anime-style art**  
            using Stable Diffusion image-to-image.

            1. Choose a **style** from the dropdown.  
            2. Capture from **webcam** or upload a photo.  
            3. (Optional) Add some extra description.  
            4. Click **Generate** and enjoy ✨
            """
        )

        with gr.Row():
            with gr.Column():
                style = gr.Dropdown(
                    label="Style",
                    choices=list(STYLE_PRESETS.keys()),
                    value="Old Vintage Photo",
                )

                extra_prompt = gr.Textbox(
                    label="Extra prompt (optional)",
                    placeholder="e.g. portrait of a young person, soft smile, studio lighting",
                    lines=2,
                )

                strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.6,
                    label="Strength (how much to change the input image)",
                )

                guidance = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=7.5,
                    label="Guidance Scale (how strongly to follow the style prompt)",
                )

                steps = gr.Slider(
                    minimum=5,
                    maximum=40,
                    step=1,
                    value=20,
                    label="Inference Steps (more = slower, higher quality)",
                )

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                input_image = gr.Image(
                    label="Webcam or Upload",
                    sources=["webcam", "upload"],  # works with your Gradio version
                    type="pil",
                )
                output_image = gr.Image(label="Stylized Output", height=448)

        generate_btn.click(
            fn=generate_from_cam,
            inputs=[style, extra_prompt, input_image, strength, guidance, steps],
            outputs=[output_image],
        )

        gr.Markdown(
            """
            ---  
            💡 **Tips**

            - On CPU only, use **lower steps (10–20)** and keep the image size as is for faster results.  
            - Try different styles on the same input for fun comparisons.  
            - Use the extra prompt to specify mood, age, lighting, etc.
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    # `share=True` if you want a public link (e.g. for demos)
    demo.launch()
