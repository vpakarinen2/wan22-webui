import gradio as gr

from app.services.wan_runner import run_i2v, cancel_current
from app.config.settings import settings


def build_i2v_tab() -> None:
    """Build the I2V tab UI."""
    with gr.Tab("I2V"):
        gr.Markdown("### Image-to-Video (Wan2.2-I2V)")
        with gr.Row():
            input_image = gr.Image(label="Reference Image", type="filepath")
            with gr.Column():
                prompt = gr.Textbox(label="Prompt (optional)")
                size = gr.Dropdown(
                    label="Resolution",
                    choices=[
                        "1280*720", "720*1280",
                        "832*480",  "480*832",
                    ],
                    value="480*832",
                )
                ckpt_dir = gr.Textbox(
                    label="Checkpoint Directory",
                    value=f"{settings.models_root}/Wan2.2-I2V-A14B",
                    placeholder="/workspace/models/Wan2.2-I2V-A14B",
                )
        with gr.Accordion("Advanced", open=False):
            with gr.Row():
                sample_steps = gr.Slider(
                    label="Sampling Steps",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=20,
                    info="Lower = faster, Higher = better detail.",
                )
                frame_num = gr.Slider(
                    label="Frame Count",
                    minimum=8,
                    maximum=129,
                    step=1,
                    value=65,
                    info="Total frames to generate.",
                )
                sample_solver = gr.Dropdown(
                    label="Sampler",
                    choices=["unipc", "dpm++"],
                    value="unipc",
                )
            with gr.Row():
                out_fps = gr.Slider(
                    label="Output FPS",
                    minimum=8,
                    maximum=30,
                    step=1,
                    value=16,
                    info="Frames per second for the output video.",
                )
                seed = gr.Number(
                    label="Seed",
                    value=0,
                    precision=0,
                    info="Use 0 for random seed.",
                )
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
        with gr.Accordion("Performance", open=False):
            with gr.Row():
                offload_model = gr.Checkbox(
                    label="Offload Model (CPU↔GPU)", value=False,
                    info="Enable to reduce VRAM at the cost of speed.",
                )
                t5_cpu = gr.Checkbox(
                    label="Run T5 on CPU", value=False,
                    info="Saves VRAM but slows text encoding.",
                )
                prefer_flash = gr.Checkbox(
                    label="Use Flash Attention", value=True,
                    info="Use FlashAttention if available.",
                )

        with gr.Row():
            run_btn = gr.Button("Generate", variant="primary")
            cancel_btn = gr.Button("Cancel", variant="stop")
        output_video = gr.Video(label="Output Video")
        logs = gr.Textbox(label="Logs", lines=6)

        eta_md = gr.Markdown("Estimated time: ~ --")

        def _parse_size(area: str) -> tuple[int, int]:
            try:
                w, h = area.split("*")
                return int(w), int(h)
            except Exception:
                return 1024, 704

        def estimate_eta(_steps: float, _frames: float, _area: str, _offload: bool, _t5cpu: bool, _prefer_flash: bool):
            w, h = _parse_size(_area)
            res_factor = max(1e-6, (w * h) / float(1024 * 704))
            per_frame_step = 0.25
            mult = 1.0
            if not _prefer_flash:
                mult *= 1.25
            if _offload:
                mult *= 1.35
            if _t5cpu:
                mult *= 1.05
            total_s = float(_steps) * float(_frames) * per_frame_step * res_factor * mult

            if total_s < 60:
                return f"Estimated time: ~ {int(total_s)}s"
            m = int(total_s // 60)
            s = int(total_s % 60)
            return f"Estimated time: ~ {m}m {s:02d}s"

        def on_run(image: str, text_prompt: str, area: str, ckpt: str, steps: float, solver: str, frames: float, _offload: bool, _t5cpu: bool, _prefer_flash: bool, _out_fps: float, _randomize: bool, _seed: float):
            if _randomize:
                seed_arg = None
            else:
                try:
                    seed_val = int(_seed)
                except Exception:
                    seed_val = 0
                seed_arg = None if seed_val is None or seed_val == 0 else seed_val
            return run_i2v(
                image=image,
                prompt=text_prompt,
                size=area,
                ckpt_dir=ckpt,
                sample_steps=steps,
                sample_solver=solver,
                frame_num=frames,
                offload_model=_offload,
                t5_cpu=_t5cpu,
                prefer_flash_attn=_prefer_flash,
                out_fps=int(_out_fps),
                seed=seed_arg,
            )

        run_btn.click(on_run, [input_image, prompt, size, ckpt_dir, sample_steps, sample_solver, frame_num, offload_model, t5_cpu, prefer_flash, out_fps, randomize_seed, seed], [output_video, logs])

        def _toggle_seed(r: bool):
            if r:
                return gr.update(interactive=False, value=0)
            return gr.update(interactive=True)

        randomize_seed.change(_toggle_seed, inputs=randomize_seed, outputs=seed)
        cancel_btn.click(lambda: cancel_current(), outputs=logs)

        for comp in (sample_steps, frame_num, size, offload_model, t5_cpu, prefer_flash):
            comp.change(
                estimate_eta,
                inputs=[sample_steps, frame_num, size, offload_model, t5_cpu, prefer_flash],
                outputs=eta_md,
            )
