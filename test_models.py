#!/usr/bin/env python3
"""Test script to check which Gemini models are available"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models to test
test_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro", 
    "gemini-pro",
    "gemini-1.0-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-pro",
]

working_models = []
failed_models = []

print("Testing Gemini models...")
print("=" * 50)

for model_name in test_models:
    try:
        print(f"Testing {model_name}...", end=" ")
        
        # Try to create the model
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            google_api_key=GEMINI_API_KEY
        )
        
        # Try a simple test
        response = llm.invoke("Hello")
        
        working_models.append(model_name)
        print("✅ WORKING")
        
    except Exception as e:
        failed_models.append(model_name)
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print("⚠️  RATE LIMITED")
        elif "404" in error_msg or "not found" in error_msg.lower():
            print("❌ NOT FOUND")
        else:
            print(f"❌ ERROR: {error_msg[:50]}...")

print("\n" + "=" * 50)
print("RESULTS:")
print(f"✅ Working models: {working_models}")
print(f"❌ Failed models: {failed_models}")

if working_models:
    print(f"\n💡 Recommended model list for your code:")
    print("GEMINI_MODELS = [")
    for model in working_models:
        print(f'    "{model}",')
    print("]")
else:
    print("\n⚠️  No working models found. You may have hit daily limits.")