from unsloth import FastVisionModel
from transformers import TextStreamer
from transformers import AutoTokenizer
import torch
from PIL import Image

# --------------------------------------------------------
# Load the fine-tuned model and tokenizer at import time
# --------------------------------------------------------
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="Johnx69/road-finetuned",  # Change to your trained model
    load_in_4bit=True,  # Set to False for full 16-bit LoRA
    device_map="cuda:0",
)
FastVisionModel.for_inference(model)  # Enable inference mode


def infer_and_return(image_path, instruction):
    """
    Run inference on an image given an instruction.
    Returns the generated text as a string.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Construct the message
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]

    # Prepare the input for the model
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
            do_sample=True
        )

    # Decode to text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text


# Example standalone usage if you ever want to test:
#
# if __name__ == "__main__":
#     image_path = "images/sample.jpg"
#     instruction = (
#       "You are an expert in road pothole evaluation. "
#       "Analyze the image and comment on its severity."
#     )
#     result = infer_and_return(image_path, instruction)
#     print("Result:", result)
