import base64
import os
from google import genai
from google.genai import types

value = os.environ.get('api_key')
print(value)

def generate():
    client = genai.Client(
        api_key=os.environ.get("api_key"),
    )

    model = "gemini-2.0-pro-exp-02-05"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
