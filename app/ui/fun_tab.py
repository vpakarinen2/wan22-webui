import gradio as gr

from app.services.fun_runner import run_fun_control
from app.config.settings import settings


def build_fun_tab() -> None:
    """Build the Fun Control tab UI."""
    with gr.Tab("Fun Control"):
        gr.Markdown("### Fun Control (Wan2.2-Fun-Control)")

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
                control_type = gr.Dropdown(
                    label="Control Type",
                    choices=["canny", "depth", "pose"],
                    value="canny",
                )
                control_strength = gr.Slider(
                    label="Control Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.7,
                    info="How strongly the control map guides generation.",
                )

        with gr.Accordion("Advanced (Control Settings)", open=False):
            with gr.Row():
                canny_low = gr.Slider(
                    label="Canny Low Threshold",
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=100,
                )
                canny_high = gr.Slider(
                    label="Canny High Threshold",
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=200,
                )
            with gr.Row():
                pose_hand = gr.Checkbox(label="Pose: Detect Hands", value=True)
                pose_face = gr.Checkbox(label="Pose: Detect Face", value=False)

        with gr.Accordion("Control Video (Optional)", open=False):
            with gr.Row():
                control_video = gr.Video(label="Control Video (optional)", sources=["upload"], include_audio=False)
                control_video.type = "filepath"
            with gr.Row():
                control_source = gr.Dropdown(
                    label="Control Source",
                    choices=[
                        "precomputed",
                        "rgb->canny",
                        "rgb->pose",
                        "rgb->depth",
                    ],
                    value="rgb->canny",
                    info="How to derive the control from the uploaded video frames.",
                )

        with gr.Accordion("Advanced (Sampling)", open=False):
            with gr.Row():
                sample_steps = gr.Slider(
                    label="Sampling Steps",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=16,
                )
                frame_num = gr.Slider(
                    label="Frame Count",
                    minimum=8,
                    maximum=129,
                    step=1,
                    value=49,
                )
                sample_solver = gr.Dropdown(
                    label="Sampler",
                    choices=["fm-euler", "unipc", "dpm"],
                    value="unipc",
                )
            with gr.Row():
                fps = gr.Slider(
                    label="Output FPS",
                    minimum=8,
                    maximum=30,
                    step=1,
                    value=16,
                    info="Frames per second of the written MP4 (e.g., 81 frames at 16 fps ≈ 5.06s)",
                )
                use_inpaint = gr.Checkbox(
                    label="Use Ref Image (Inpaint)",
                    value=False,
                    info="Keeps more of the input image (less motion).",
                )
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.0,
                    maximum=12.0,
                    step=0.1,
                    value=6.0,
                    info="Classifier-free guidance.",
                )

        with gr.Row():
            run_btn = gr.Button("Generate", variant="primary")
        output_video = gr.Video(label="Output Video")
        logs = gr.Textbox(label="Logs", lines=6)

        eta_md = gr.Markdown("Estimated time: ~ --")

        def _parse_size(area: str) -> tuple[int, int]:
            try:
                w, h = area.split("*")
                return int(w), int(h)
            except Exception:
                return 1024, 704

        def estimate_eta(_steps: float, _frames: float, _area: str, _ctype: str):
            w, h = _parse_size(_area)
            res_factor = max(1e-6, (w * h) / float(1024 * 704))
            base = 0.24
            if _ctype == "depth":
                base = 0.27
            elif _ctype == "pose":
                base = 0.28
            total_s = float(_steps) * float(_frames) * base * res_factor
            if total_s < 60:
                return f"Estimated time: ~ {int(total_s)}s"
            m = int(total_s // 60)
            s = int(total_s % 60)
            return f"Estimated time: ~ {m}m {s:02d}s"

        def on_run(
            image: str,
            text_prompt: str,
            area: str,
            ctype: str,
            cstrength: float,
            c_low: float,
            c_high: float,
            p_hand: bool,
            p_face: bool,
            steps: float,
            frames: float,
            solver: str,
            gscale: float,
            _use_inpaint: bool,
            out_fps: float,
            ctrl_video_path: str | None,
            ctrl_source: str,
        ):
            return run_fun_control(
                image_path=image,
                save_path=f"{settings.outputs_root}/fun_control_out.mp4",
                size=area,
                prompt=text_prompt,
                control_type=ctype,
                control_strength=cstrength,
                canny_low_high=(int(c_low), int(c_high)),
                pose_hand=p_hand,
                pose_face=p_face,
                sample_steps=int(steps),
                frame_num=int(frames),
                sample_solver=solver,
                guidance_scale=float(gscale),
                use_inpaint=bool(_use_inpaint),
                fps=int(out_fps),
                control_video_path=ctrl_video_path,
                control_source=ctrl_source,
            )

        run_btn.click(
            on_run,
            [
                input_image,
                prompt,
                size,
                control_type,
                control_strength,
                canny_low,
                canny_high,
                pose_hand,
                pose_face,
                sample_steps,
                frame_num,
                sample_solver,
                guidance_scale,
                use_inpaint,
                fps,
                control_video,
                control_source,
            ],
            [output_video, logs],
        )

        for comp in (sample_steps, frame_num, size, control_type):
            comp.change(
                estimate_eta,
                inputs=[sample_steps, frame_num, size, control_type],
                outputs=eta_md,
            )
