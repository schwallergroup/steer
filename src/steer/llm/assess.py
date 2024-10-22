"""Evaluate a given reaction."""

import os
from openai import OpenAI
from prompts import system_prompt
import base64
from dotenv import load_dotenv

load_dotenv()


client = OpenAI()

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

img_path = "src/steer/llm/test_img.png"
b64img = encode_image(img_path)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the image of the reaction:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64img}"},
                },
            ],
        }
    ],
)

if __name__ == "__main__":
    print(response.choices[0].message.content)