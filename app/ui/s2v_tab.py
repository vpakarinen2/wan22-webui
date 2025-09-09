import gradio as gr

from app.services.wan_runner import run_s2v, cancel_current
from app.config.settings import settings


def build_s2v_tab() -> None:
    """Build the S2V tab UI."""
    with gr.Tab("S2V"):
        gr.Markdown("### Sound-to-Video (Wan2.2-S2V)")
        with gr.Row():
            ref_image = gr.Image(label="Reference Image", type="filepath")
            audio = gr.Audio(label="Audio (upload or record)", type="filepath")
        with gr.Row():
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
                value=f"{settings.models_root}/Wan2.2-S2V-14B",
                placeholder="/workspace/models/Wan2.2-S2V-14B",
            )
        with gr.Accordion("Advanced", open=False):
            with gr.Row():
                sample_steps = gr.Slider(
                    label="Sampling Steps",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=20,
                    info="Lower = faster, Higher = more detail.",
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
                    info="Use FlashAttention when available.",
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
            per_frame_step = 0.27
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

        def on_run(
            _ref_image: str,
            _audio: str,
            _prompt: str,
            _size: str,
            _ckpt_dir: str,
            _sample_steps: float,
            _sample_solver: str,
            _frame_num: float,
            _offload_model: bool,
            _t5_cpu: bool,
            _prefer_flash: bool,
        ):
            return run_s2v(
                ref_image=_ref_image,
                audio=_audio,
                prompt=_prompt,
                size=_size,
                ckpt_dir=_ckpt_dir,
                sample_steps=_sample_steps,
                sample_solver=_sample_solver,
                frame_num=_frame_num,
                offload_model=_offload_model,
                t5_cpu=_t5_cpu,
                prefer_flash_attn=_prefer_flash,
            )

        run_btn.click(
            on_run,
            [ref_image, audio, prompt, size, ckpt_dir, sample_steps, sample_solver, frame_num, offload_model, t5_cpu, prefer_flash],
            [output_video, logs],
        )
        cancel_btn.click(lambda: cancel_current(), outputs=logs)

        for comp in (sample_steps, frame_num, size, offload_model, t5_cpu, prefer_flash):
            comp.change(
                estimate_eta,
                inputs=[sample_steps, frame_num, size, offload_model, t5_cpu, prefer_flash],
                outputs=eta_md,
            )
