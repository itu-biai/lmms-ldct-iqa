# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 11:36:29 2025

@author: kagan
"""

import os
import re
import tifffile as tiff
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI

client = OpenAI(
    api_key="yourkey",
    base_url="https://openrouter.ai/api/v1"
)

model_name = "meta-llama/llama-4-maverick"

image_folder = "ımagepath"
output_path = os.path.join("yourpath", "Meta Llama4 API Scores Zero Shot.txt")

def tifftojpeg(image_path):
    img = tiff.imread(image_path)
    img = 255 * (img - img.min()) / (img.max() - img.min() + 1e-5)
    img = img.astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def imagescoring(image_path):
    base64_image = tifftojpeg(image_path)

    prompt = (
            "You are an expert radiologist performing image quality control of abdominal CT scans. "
    "Your task is strictly to assess the **technical image quality**, NOT any medical diagnosis.\n\n"

    "Evaluate the image based on:\n"
    "- Diagnostic usability (organ visibility: liver, kidneys, bowel, spine)\n"
    "- Noise (graininess, detail obscuration)\n"
    "- Artifacts (streaks, distortions)\n"
    "- Contrast (soft tissue boundary clarity)\n\n"

    "Now give a score between 0.0 and 4.0 **as a float** (e.g., 1.3, 2.4, 3.1, 3.7).\n"
    "**Avoid repeating scores too often.** Do not use the same values across multiple images unless truly justified.\n"
    "Avoid defaulting to 3.0, 3.2, 3.5 unless they exactly reflect the image's balance.\n"
    "**Be diverse and realistic** in your scoring: consider using values like 1.8, 2.6, 2.9, 3.4, etc.\n\n"
    "Never skip scoring. Always provide a score and explanation even if the image is difficult to interpret.\n\n"
    "⚠️ VERY IMPORTANT SCORING RULES:\n"
        "- Do NOT reuse scores like 2.1, 2.7, or 3.0 too often.\n"
        "- Always choose a **unique score** that reflects the image's own quality.\n"
        "- The model will be penalized if it repeats the same float score.\n"
        "- DO NOT CHOOSE ALWAYS SAME FLOAT NUMBER FOR EACH IMAGE! THIS IS A STRICT RULE"
        "- Randomize slightly when uncertain — pick diverse but realistic values like 1.7, 2.4, 3.3, etc.\n"
    "### Output format:\n"
    "Score (float): X.X\n"
    "Category: Bad / Poor / Fair / Good / Excellent\n"
    "Explanation: Write 2–3 professional sentences explaining the score. Discuss organ visibility, noise, artifact presence, and contrast clarity.\n\n"
    "Be detailed, fair, and consistent — use your professional judgment."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=2000,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error evaluating image: {e}"

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("Meta API Scores Zero Shot\n")

    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.tif'):
            print(f"\nEvaluating {filename}...")
            image_path = os.path.join(image_folder, filename)
            result = imagescoring(image_path)
            print(result)

            score_match = re.search(r"Score \(float\):\s*([\d.]+)", result)
            category_match = re.search(r"Category:\s*(\w+)", result)

            if score_match and category_match:
                score = score_match.group(1)
                category = category_match.group(1)
                f.write(f'"{filename}": {score}, {category}\n')
            else:
                f.write(f'"{filename}": Error parsing score/category\n')