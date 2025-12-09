
import google.generativeai as genai
import os

try:
    api_key = ""
    with open(".streamlit/secrets.toml", "r") as f:
        for line in f:
            if "GEMINI_API_KEY" in line:
                # Naive parse: GEMINI_API_KEY = "xyz"
                parts = line.split("=")
                if len(parts) >= 2:
                    api_key = parts[1].strip().strip('"').strip("'")
                    break
    
    if not api_key:
        print("Could not find API Key")
        exit(1)

    genai.configure(api_key=api_key)
    
    print("Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
