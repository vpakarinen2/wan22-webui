import gradio as gr

from app.ui.i2v_tab import build_i2v_tab
from app.ui.s2v_tab import build_s2v_tab
#from app.ui.fun_tab import build_fun_tab


def create_app() -> gr.Blocks:
    """Create the Gradio app with tabs for I2V and S2V."""
    with gr.Blocks(title="Wan2.2 WebUI") as wangd:
        gr.HTML(
            """
        <h1 style="text-align:center; margin: 12px 0; font-size: 32px; font-weight: 700;">
            Wan2.2 WebUI (I2V + S2V)
        </h1>
            """
        )
        with gr.Tabs():
            build_i2v_tab()
            build_s2v_tab()
    return wangd
