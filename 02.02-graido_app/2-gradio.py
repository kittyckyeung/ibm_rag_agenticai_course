import gradio as gr

def add_strings(Str1, Str2):
    # Ensure inputs are strings and handle None
    return str(Str1 or '') + str(Str2 or '')

# Define the interface
demo = gr.Interface(
    fn=add_strings,
    inputs=[gr.Textbox(label="String 1"), gr.Textbox(label="String 2")],
    outputs=gr.Textbox(label="Result")
)

# Launch the interface
demo.launch(server_name="127.0.0.1", server_port= 7860)
