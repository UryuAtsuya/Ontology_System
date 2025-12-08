import google.generativeai as genai
import os

api_key = os.environ.get("GOOGLE_API_KEY")
print(f"Key: {api_key[:5]}...")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")
try:
    resp = model.generate_content("Hello")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
