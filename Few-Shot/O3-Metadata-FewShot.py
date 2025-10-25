# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 20:25:46 2025

@author: kagan
"""

import re
import random
from openai import OpenAI


client = OpenAI(
    api_key="yourkey"
)

model_name = "o3"


train_region_path = "yourpath\Region Labels Train.txt"
test_region_path = "yourpath\Region Labels Test.txt"
train_score_path = "yourpath\Radiolog Scores Few Shot Training.txt"
train_noise_path = "yourpath\Noise Train Final.txt"
test_noise_path = "yourpath\Noise Test Final.txt"
output_path = r"C:\Users\kagan\OneDrive\Masaüstü\Few Shot Results\O3-Metadata-API Scores Few Shot.txt"

def load_region_labels(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r"(.+?):\s*(https?://.+?),\s*(\w+)", line.strip())
            if match:
                filename = match.group(1).strip()
                url = match.group(2).strip()
                region = match.group(3).strip().lower()
                data[filename] = {"url": url, "region": region}
    return data

def load_scores(path):
    scores = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'"(.+?)\.tif":\s*([\d.]+),\s*(\w+)', line.strip())
            if match:
                filename = match.group(1) + ".png"
                score = float(match.group(2))
                category = match.group(3)
                scores[filename] = (score, category)
    return scores

def load_noise(path):
    noise = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'"(.+?)":\s*([\d.]+)', line.strip())
            if match:
                noise[match.group(1).strip()] = float(match.group(2))
    return noise

def fewshotmessages(train_data, train_scores, train_noise, region, max_count=34):
    candidates = [
        (fname, train_data[fname]["url"], train_scores[fname], train_noise.get(fname.replace(".png", ".tif"), None))
        for fname in train_data
        if train_data[fname]["region"] == region and fname in train_scores
    ]
    if not candidates:
        candidates = [
            (fname, train_data[fname]["url"], train_scores[fname], train_noise.get(fname.replace(".png", ".tif"), None))
            for fname in train_data
            if fname in train_scores
        ]
    random.shuffle(candidates)
    selected = candidates[:max_count]

    messages = []
    for fname, url, (score, category), noise in selected:
        noise_text = f"Estimated noise level: {noise:.3f}" if noise is not None else "Noise level: Unknown"
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"Evaluate this CT scan image.\n{noise_text}"},
            {"type": "image_url", "image_url": {"url": url}}
        ]})
        messages.append({"role": "assistant", "content": (
            f"Score (float): {score}\n"
            f"Category: {category}\n"
            f"Explanation: This is a reference scan labeled by a radiologist. "
            f"The image quality corresponds to a {category} level, with appropriate visibility, noise, artifacts, and contrast for this score."
        )})
    return messages

def imagescoring(test_url, test_noise, few_shot_messages):
    prompt = (
        f"You are now shown a new CT scan image. Estimated noise level: {test_noise:.3f}.\n"
        "Based on the examples above, evaluate its **technical image quality \n\n"
        "Use the same tone, criteria, and evaluation style as in the few-shot examples.\n"
        "Assess the following:\n"
        "- Diagnostic usability (Are key organs like liver, kidneys, bowel, spine visible and clear?)\n"
        "- Noise (Is there graininess or loss of detail?)\n"
        "- Artifacts (Are there any streaks, motion blur, or distortions?)\n"
        "- Contrast (Are soft tissue boundaries clearly distinguishable?)\n\n"
        "Rate the **overall image quality** strictly using a **float number between 0.0 and 4.0**.\n"
        "Be diverse and precise. Avoid repeating values too often.\n"
        "Output format:\n"
        "Score (float): X.X\nCategory: Bad / Poor / Fair / Good / Excellent\nExplanation: ...\n"
        "Only return 2 lines as output."
    )

    messages = few_shot_messages + [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": test_url}}
        ]
    }]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


train_data = load_region_labels(train_region_path)
test_data = load_region_labels(test_region_path)
train_scores = load_scores(train_score_path)
train_noise = load_noise(train_noise_path)
test_noise = load_noise(test_noise_path)

with open(output_path, 'w', encoding='utf-8') as outfile:
    outfile.write("O3-LABEL RESULTS WITH NOISE AWARENESS\n")

    retry_list = []

    for i, (filename, info) in enumerate(test_data.items()):
        url = info["url"]
        region = info["region"]
        noise_level = test_noise.get(filename.replace(".png", ".tif"), 0.0)
        few_shots = fewshotmessages(train_data, train_scores, train_noise, region, max_count=34)

        print(f"\n[{i+1}/{len(test_data)}] Evaluating {filename} (Region: {region}, Noise: {noise_level:.3f})...")
        result = imagescoring(url, noise_level, few_shots)
        print(result)

        score_match = re.search(r"Score \(float\):\s*([\d.]+)", result)
        category_match = re.search(r"Category:\s*(\w+)", result)
        if score_match and category_match:
            score = score_match.group(1)
            category = category_match.group(1)
            outfile.write(f'"{filename.replace(".png", ".tif")}": {score}, {category}\n')
        else:
            retry_list.append((filename, url, region, noise_level))


if retry_list:
    print(f"\nRetrying {len(retry_list)} failed cases...\n")
    with open(output_path, 'a', encoding='utf-8') as outfile:
        for attempt in range(5):  
            still_failed = []
            for filename, url, region, noise_level in retry_list:
                few_shots = fewshotmessages(train_data, train_scores, train_noise, region, max_count=34)

                print(f"Retrying {filename} (attempt {attempt+1})...")
                result = imagescoring(url, noise_level, few_shots)
                print(result)

                score_match = re.search(r"Score \(float\):\s*([\d.]+)", result)
                category_match = re.search(r"Category:\s*(\w+)", result)
                if score_match and category_match:
                    score = score_match.group(1)
                    category = category_match.group(1)
                    outfile.write(f'"{filename.replace(".png", ".tif")}": {score}, {category}\n')
                else:
                    still_failed.append((filename, url, region, noise_level))
            retry_list = still_failed
            if not retry_list:
                break

    if retry_list:
        print(f"\n Final failed cases: {len(retry_list)} — Could not be parsed after retries.")
        with open(output_path, 'a', encoding='utf-8') as outfile:
            for filename, _, _, _ in retry_list:
                outfile.write(f'"{filename.replace(".png", ".tif")}": ERROR - final retry failed\n')
