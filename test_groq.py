import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

client = Groq(api_key=api_key)

model_name = "openai/gpt-oss-120b"

print("Testing Groq model...")
print("Model:", model_name)

try:

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Say hello in one sentence."
            }
        ],
        temperature=0
    )

    print("\n✅ Groq model is working!\n")

    print(
        response.choices[0].message.content
    )

except Exception as e:

    print("\n❌ Groq model test failed!")
    print(type(e).__name__)
    print(e)