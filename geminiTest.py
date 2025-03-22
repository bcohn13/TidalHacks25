from google import genai
import os

value = os.environ.get('api_key')
print(value)
client = genai.Client(api_key=value)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)

print(response.text)