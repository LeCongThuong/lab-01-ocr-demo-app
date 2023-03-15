import gradio as gr
from util import ocr

gr.Interface(fn=ocr,
             inputs=gr.Image(type="pil"),
             outputs=gr.Textbox(),
             allow_flagging="never",
             examples=["demo.png"]
             ).launch(share=True)
