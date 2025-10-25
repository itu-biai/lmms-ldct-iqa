# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:03:33 2025

@author: kagan
"""
import re
from openai import OpenAI

client = OpenAI(
    api_key="yourkey",
    base_url="https://openrouter.ai/api/v1"
)

model_name = "openai/o3"

train_score_path = "yourpathRadiolog Scores Few Shot Training.txt"
train_url_path = r"C:\Users\kagan\OneDrive\Masaüstü\Few Shot Training.txt"
test_url_path = r"C:\Users\kagan\OneDrive\Masaüstü\Few Shot Testing.txt"
output_path = "yourpath/O3-5 API Scores Few Shot.txt"

def load_scores(path):
    scores = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'"(.+?)\.tif":\s*([\d.]+),\s*(\w+)', line.strip())
            if match:
                filename = match.group(1) + ".png"
                scores[filename] = (float(match.group(2)), match.group(3))
    return scores

def load_urls(path):
    urls = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ":" in line:
                parts = line.strip().split(":", 1)
                filename = parts[0].strip()
                url = parts[1].strip()
                urls[filename] = url
    return urls

def fewshotmessages(train_scores, train_urls, start_idx=0, count=10):
    messages = []
    items = list(train_scores.items())[start_idx:start_idx + count]
    for filename, (score, category) in items:
        if filename in train_urls:
            url = train_urls[filename]
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Evaluate this CT scan image."},
                {"type": "image_url", "image_url": {"url": url}}
            ]})
            messages.append({"role": "assistant", "content": (
                f"Score (float): {score}\n"
                f"Category: {category}\n"
                f"Explanation: This is a reference scan labeled by a radiologist. "
                f"The image quality corresponds to a {category} level, with appropriate visibility, noise, artifacts, and contrast for this score."
            )})
    return messages

def imagescoring(test_url, few_shot_messages):
    test_prompt = (
        "You are now shown a new abdominal CT scan image. Based on the examples above, evaluate its **technical image quality only**, not any medical condition.\n\n"
        "Use the same tone, criteria, and evaluation style as in the few-shot examples.\n"
        "Assess the following:\n"
        "- Diagnostic usability (Are key organs like liver, kidneys, bowel, spine visible and clear?)\n"
        "- Noise (Is there graininess or loss of detail?)\n"
        "- Artifacts (Are there any streaks, motion blur, or distortions?)\n"
        "- Contrast (Are soft tissue boundaries clearly distinguishable?)\n\n"
        "Rate the **overall image quality** strictly using a **float number between 0.0 and 4.0**, where 0.0 is the lowest possible and 4.0 is the maximum allowed. Do not exceed this range.\n"
        "Examples: 0.8, 1.5, 2.4, 3.2, 3.9, 4.0\n"
        "**Do not repeat scores too frequently. Avoid defaulting to 3.0, 3.2, or 3.5. Be diverse and precise.**\n"
        "**Never return a value above 4.0. Never skip scoring. Always include a float score.**\n\n"
        "⚠️ VERY IMPORTANT SCORING RULES:\n"
        "- Do NOT reuse scores like 2.1, 2.7, or 3.0 too often.\n"
        "- Always choose a **unique score** that reflects the image's own quality.\n"
        "- The model will be penalized if it repeats the same float score.\n"
        "- DO NOT CHOOSE ALWAYS SAME FLOAT NUMBER FOR EACH IMAGE! THIS IS A STRICT RULE"
        "- Randomize slightly when uncertain — pick diverse but realistic values like 1.7, 2.4, 3.3, etc.\n"
        "### Output format:\n"
        "Score (float): X.X (from 0.0 to 4.0 only)\n"
        "Category: Bad / Poor / Fair / Good / Excellent\n"
        "Explanation: In 2–3 professional sentences, explain why you gave this score. Mention organ visibility, noise level, presence of artifacts, and clarity of contrast.\n"
        "Do NOT write anything else. Do NOT explain. Just score and category.\n"
        "Your output MUST be exactly 2 lines.\n"
        "Begin now."
    )

    messages = few_shot_messages + [{
        "role": "user",
        "content": [
            {"type": "text", "text": test_prompt},
            {"type": "image_url", "image_url": {"url": test_url}}
        ]
    }]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

train_scores = load_scores(train_score_path)
train_urls = load_urls(train_url_path)
test_urls = load_urls(test_url_path)
train_keys = list(train_scores.keys())
num_train = len(train_keys)
train_batch_size = 5
retry_list = []

with open(output_path, 'w', encoding='utf-8') as outfile:
    outfile.write("O3-5 API Scores Few Shots Testing\n")

    for i, (filename, url) in enumerate(test_urls.items()):
        train_start_idx = (i * train_batch_size) % num_train
        few_shot_messages = fewshotmessages(train_scores, train_urls, start_idx=train_start_idx, count=train_batch_size)

        print(f"\nEvaluating {filename} (train batch index {train_start_idx})...")
        result = imagescoring(url, few_shot_messages)
        print(result)

        score_match = re.search(r"Score \(float\):\s*([\d.]+)", result)
        category_match = re.search(r"Category:\s*(\w+)", result)

        if score_match and category_match:
            score = score_match.group(1)
            category = category_match.group(1)
            outfile.write(f'"{filename.replace(".png", ".tif")}": {score}, {category}\n')
        else:
            retry_list.append((i, filename, url))

if retry_list:
    print(f"\nRetrying {len(retry_list)} failed cases...\n")
    with open(output_path, 'a', encoding='utf-8') as outfile:
        for i, filename, url in retry_list:
            attempts = 0
            parsed = False
            while attempts < 10 and not parsed:
                train_start_idx = (i * train_batch_size) % num_train
                few_shot_messages = fewshotmessages(train_scores, train_urls, start_idx=train_start_idx, count=train_batch_size)

                print(f"\nRetrying {filename} (attempt {attempts + 1})...")
                result = imagescoring(url, few_shot_messages)
                print(result)

                score_match = re.search(r"Score \(float\):\s*([\d.]+)", result)
                category_match = re.search(r"Category:\s*(\w+)", result)

                if score_match and category_match:
                    score = score_match.group(1)
                    category = category_match.group(1)
                    outfile.write(f'"{filename.replace(".png", ".tif")}": {score}, {category}\n')
                    parsed = True
                else:
                    attempts += 1

            if not parsed:
                outfile.write(f'"{filename.replace(".png", ".tif")}": Final error in parsing result\n')

with open(output_path, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

header = lines[0]
valid_lines = []

for line in lines[1:]:
    score_match = re.match(r'"(.+?\.tif)":\s*([\d.]+),\s*(\w+)', line.strip())
    if score_match:
        valid_lines.append(line)

with open(output_path, 'w', encoding='utf-8') as outfile:
    outfile.write(header)
    outfile.writelines(valid_lines)
