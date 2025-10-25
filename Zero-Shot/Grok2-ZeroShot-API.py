# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:08:16 2025

@author: kagan
"""

import os
import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import tifffile as tiff
import numpy as np
import re

client = OpenAI(
    api_key="yourkey",
    base_url="https://openrouter.ai/api/v1"
)

model_name = "x-ai/grok-2-vision-1212"
image_folder = "ımagepath"
output_path = os.path.join("yourpath", "Grok2 API Scores Zero Shot.txt")


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
        "You are an expert radiologist performing technical quality control of abdominal CT scans.\n\n"
        
        "Your task is to strictly evaluate the **technical image quality only**, not any medical diagnosis or clinical finding.\n\n"
        
        "Evaluate the image based on the following criteria:\n"
        "- Diagnostic usability: Are key organs (liver, kidneys, bowel, spine) clearly visible?\n"
        "- Noise: Is there graininess or detail loss that affects interpretation?\n"
        "- Artifacts: Are there motion blur, streaks, or distortions?\n"
        "- Contrast: Are soft tissue boundaries distinguishable?\n\n"
        
        "Give a **float score from 0.0 to 4.0**, where:\n"
        "- 0.0 = unusable image\n"
        "- 4.0 = excellent technical quality\n"
        "Examples of valid scores: 0.9, 1.6, 2.3, 2.9, 3.4, 3.8\n\n"
        
        "⚠️ VERY IMPORTANT SCORING RULES:\n"
        "- Do NOT reuse scores like 2.1, 2.7, or 3.0 too often.\n"
        "- Always choose a **unique score** that reflects the image's own quality.\n"
        "- The model will be penalized if it repeats the same float score.\n"
        "- DO NOT CHOOSE ALWAYS SAME FLOAT NUMBER FOR EACH IMAGE! THIS IS A STRICT RULE"
        "- Randomize slightly when uncertain — pick diverse but realistic values like 1.7, 2.4, 3.3, etc.\n"
        
        "NEVER return a value above 4.0.\n"
        "NEVER skip scoring.\n\n"
        
        "### Output format:\n"
        "Score (float): X.X\n"
        "Category: Bad / Poor / Fair / Good / Excellent\n"
        "Explanation: In 2–3 professional sentences, explain your score. Discuss visibility of organs, noise level, artifact presence, and contrast.\n\n"
        
        "Output ONLY the score, category, and explanation. Do NOT include anything else.\n"
        "Begin now."
    )



    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000,
            temperature=0.5
        )

        content = response.choices[0].message.content

        if "I'm unable" in content or "I can't evaluate" in content:
            return "Score (float): 0.0\nCategory: Bad\nExplanation: Unable to evaluate due to low visibility or model limitations."
        return content

    except Exception as e:
        return f"Error evaluating image: {e}"


for filename in os.listdir(image_folder):
    if filename.lower().endswith('.tif'):
        image_path = os.path.join(image_folder, filename)
        print(f"\nEvaluating {filename}...")
        result = imagescoring(image_path)
        print(result)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("Grok2 API Scores Zero Shot\n")

    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.tif'):
            image_path = os.path.join(image_folder, filename)
            result = imagescoring(image_path)
            score_match = re.search(r"Score \(float\):\s*([\d.]+)", result)
            category_match = re.search(r"Category:\s*(\w+)", result)

            if score_match and category_match:
                score = score_match.group(1)
                category = category_match.group(1)
                f.write(f'"{filename}": {score}, {category}\n')
            else:
                f.write(f'"{filename}": Error parsing score/category\n')
