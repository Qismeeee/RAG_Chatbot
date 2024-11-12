import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_gpt_api(prompt, model="gpt-4-turbo", max_tokens=100):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("Lỗi khi gọi GPT API:", e)
        return None


prompt = "Xin chào, bạn có thể giúp tôi không?"
response = call_gpt_api(prompt)
print("GPT Response:", response)
