import gradio as gr
from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image
import torch
import time
import os
from typing import Iterator, List, Dict, Any

# Custom streamer that yields tokens for Gradio streaming
class GradioStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.token_queue = []
        self.text_queue = []
        self.current_text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Called when a token is finalized
        # For debugging: print(f"Finalized text: {text}")
        self.current_text += text
        self.text_queue.append(text)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.text_queue:
            raise StopIteration
        return self.text_queue.pop(0)

# Load the fine-tuned model and tokenizer
class RoadAssessmentModel:
    def __init__(self, model_name="Johnx69/road-finetuned", device="cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.streamer = None
        self.load_model()

    def load_model(self):
        print(f"Loading model {self.model_name}...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.model_name,
            load_in_4bit=True,
            device_map=self.device,
        )
        FastVisionModel.for_inference(self.model)
        print("Model loaded successfully!")

    def generate_response(self, image, instruction) -> Iterator[str]:
        """
        Generate a response given an image and instruction, yielding tokens for streaming.
        """
        if isinstance(image, str):
            # If image is a file path
            image_pil = Image.open(image).convert("RGB")
        else:
            # If image is already a PIL image
            image_pil = image.convert("RGB")

        # Construct the message
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            }
        ]

        # Tokenize input
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.tokenizer(
            image_pil,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        # Create streamer for token-by-token generation
        self.streamer = GradioStreamer(self.tokenizer, skip_prompt=True)
        
        # Start generation in a non-blocking way
        self.model.generate(
            **inputs,
            streamer=self.streamer,
            max_new_tokens=500,
            use_cache=True,
            temperature=1.5,
            min_p=0.1
        )
        
        # Yield tokens as they are generated
        for token in self.streamer:
            yield token

# Initialize the model
road_model = RoadAssessmentModel()

# Chat history structure
def format_history(history: List[List[str]]) -> List[Dict[str, Any]]:
    formatted = []
    for exchange in history:
        user_msg, assistant_msg = exchange
        formatted.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Some messages might not have responses yet
            formatted.append({"role": "assistant", "content": assistant_msg})
    return formatted

# Define Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🛣️ Road Assessment AI
        
        Upload an image of a road and ask questions about it. The AI will analyze the image and provide a detailed assessment.
        
        ## Features
        - Detect potholes and road damage
        - Assess severity and safety risks
        - Get repair recommendations
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Road Image")
            
            with gr.Accordion("Advanced Options", open=False):
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.5, step=0.1, label="Temperature")
                max_tokens = gr.Slider(minimum=100, maximum=1000, value=500, step=50, label="Max Tokens")
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=500, 
                show_label=False,
                avatar_images=["👤", "🤖"],
                bubble_full_width=False,
            )
            
            with gr.Row():
                message_input = gr.Textbox(
                    placeholder="Ask about the road condition, potholes, or safety assessment...", 
                    label="Your Question",
                    scale=5
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
    
    # Example templates
    with gr.Accordion("Example Questions", open=True):
        example_questions = [
            "Identify any potholes in this image and describe their severity.",
            "What's the overall condition of this road and is it safe to drive on?",
            "Are there any immediate repairs needed based on what you see?",
            "How would you prioritize repairs for this road section?",
            "What types of vehicle damage could result from driving on this road?"
        ]
        
        for question in example_questions:
            gr.Button(question).click(
                lambda q: q, 
                inputs=[gr.Textbox(value=question, visible=False)], 
                outputs=[message_input]
            )
    
    # Clear button
    clear_btn = gr.Button("Clear Conversation")
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    # State management
    state = gr.State([])
    
    def user_input(user_message, image, history, chat_state):
        if not user_message.strip() and image is None:
            return gr.update(), history, chat_state
            
        # Append user message to history
        if image is not None:
            display_message = f"[Image uploaded] {user_message}"
        else:
            display_message = user_message
            
        chat_state.append({"role": "user", "message": user_message, "image": image})
        history.append([display_message, None])
        return "", history, chat_state
    
    def bot_response(history, chat_state):
        if not chat_state:
            return history
            
        last_user_interaction = chat_state[-1]
        user_message = last_user_interaction["message"]
        image = last_user_interaction["image"]
        
        if image is None:
            # Handle text-only input with a default response
            history[-1][1] = "Please upload an image for me to analyze."
            return history
            
        # Stream the response
        history[-1][1] = ""  # Initialize empty response
        instruction = user_message if user_message else "Analyze this road image and describe any damage or issues you can see."
        
        for token in road_model.generate_response(image, instruction):
            history[-1][1] += token
            yield history
            time.sleep(0.01)  # Small delay for smoother streaming
        
        return history
    
    # Set up the event chain
    submit_btn.click(
        user_input, 
        inputs=[message_input, image_input, chatbot, state], 
        outputs=[message_input, chatbot, state]
    ).then(
        bot_response, 
        inputs=[chatbot, state], 
        outputs=[chatbot]
    )
    
    # Also trigger on pressing Enter
    message_input.submit(
        user_input, 
        inputs=[message_input, image_input, chatbot, state], 
        outputs=[message_input, chatbot, state]
    ).then(
        bot_response, 
        inputs=[chatbot, state], 
        outputs=[chatbot]
    )
    
    # Example usage section
    gr.Markdown(
        """
        ## How to Use
        
        1. Upload an image of a road section
        2. Ask specific questions about road conditions, potholes, etc.
        3. Get real-time analysis from the AI
        
        This application uses a fine-tuned vision model specifically trained for road assessment.
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.queue()  # Enable queuing for faster responses
    demo.launch(server_port=8000, share=False, server_name="0.0.0.0")  # Launch on port 8000