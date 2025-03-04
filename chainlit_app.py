import chainlit as cl
from chainlit import user_session
from infer import infer_and_return


@cl.on_chat_start
async def on_chat_start():
    # Initialize message history
    user_session.set("MESSAGE_HISTORY", [])


@cl.on_message
async def main(message: cl.Message):
    """
    This function is triggered each time the user sends a message.
    We check for an image; if present, pass it along with the text
    to your locally fine-tuned model. Otherwise, just respond with a
    quick message or handle text-only mode as desired.
    """
    # Prepare a "placeholder" message to update later
    msg = cl.Message(content="", author="Road Pothole Assistant")
    await msg.send()

    # Extract any attached images
    images = [file for file in message.elements if "image" in file.mime]
    user_prompt = message.content

    # Retrieve message history from session
    message_history = user_session.get("MESSAGE_HISTORY")

    # If user attached an image, run local inference
    if len(images) > 0:
        image_path = images[0].path
        ai_response = infer_and_return(image_path, user_prompt)
    else:
        ai_response = (
            "No image detected. Please attach a road image for pothole analysis."
        )

    # Update the placeholder with the final AI response
    await msg.update(content=ai_response, author="Assistant")

    # Append user and assistant messages to the history
    message_history.append({"role": "user", "content": user_prompt})
    message_history.append({"role": "assistant", "content": ai_response})
    user_session.set("MESSAGE_HISTORY", message_history)
