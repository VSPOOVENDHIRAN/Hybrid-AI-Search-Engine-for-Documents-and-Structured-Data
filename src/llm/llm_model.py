from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_llm_response(prompt):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # ✅ FIXED
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content