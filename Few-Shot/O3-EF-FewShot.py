# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 12:43:55 2025

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
output_path = "yourpath\O3 API Scores EF.txt"

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

def fewshotmessages_with_prediction_feedback(train_data, train_scores, train_noise, region, batch_size=5):
    candidates = [
        (fname, train_data[fname]["url"], train_scores[fname], train_noise.get(fname.replace(".png", ".tif"), None))
        for fname in train_data if train_data[fname]["region"] == region and fname in train_scores
    ]
    random.shuffle(candidates)
    selected = candidates[:batch_size]

    messages = []
    error_descriptions = []

    for fname, url, (true_score, category), noise in selected:
        prompt = (
            f"You are now shown a new CT scan image. Estimated noise level: {noise:.3f}.\n"
            " Evaluate its **technical image quality\n"
            "Pay **particular attention to the provided noise level** \n"
            "If the noise level is low,the quality will be low and you will tend to give lower scores (for example, if the noise level is 0.008, this means high noise and low quality;\n"
            "if the noise level is 0.001, this means low noise and high quality and you will tend to give higher scores).\n"
            "However, balance it by taking into account all criteria, not just noise. So, you don't have to give a high score like 3.6 just because the noise is 0.001. What matters is quality.\n"
            "Assess the following:\n"
            "- Diagnostic usability (Are key organs like liver, kidneys, bowel, spine visible and clear?)\n"
            "- Noise (Is there graininess or loss of detail?)\n"
            "- Artifacts (Are there any streaks, motion blur, or distortions?)\n"
            "- Contrast (Are soft tissue boundaries clearly distinguishable?)\n\n"
            "Rate the **overall image quality** strictly using a **float number between 0.0 and 4.0**.\n"
            "Be diverse and precise. Avoid repeating values too often.\n"
            "Output format:\n"
            "Score (float): X.X\nCategory: Bad / Poor / Fair / Good / Excellent\n"

        )
        messages_batch = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}}
            ]}
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages_batch,
            max_completion_tokens=1000,
        )
        content = response.choices[0].message.content

        pred_match = re.search(r"Score \(float\):\s*([\d.]+)", content)
        pred_score = float(pred_match.group(1)) if pred_match else 0.0
        error = abs(pred_score - true_score)

        error_descriptions.append(f"True: {true_score:.1f}, Pred: {pred_score:.1f}, Error: {error:.2f}")

        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"Evaluate this CT scan.\nNoise: {noise:.3f}"},
            {"type": "image_url", "image_url": {"url": url}}
        ]})
        messages.append({"role": "assistant", "content": (
            f"Score (float): {true_score}\n"
            f"Category: {category}\nExplanation: Radiologist label."
        )})
        print(f"Training example: {fname}")
        print(f"  True score     : {true_score}")
        print(f"  Predicted score: {pred_score}")
        print(f"  Error          : {error:.2f}")
        print("-" * 50)

    feedback_text = "\n".join([f"Example {i+1}: {desc}" for i, desc in enumerate(error_descriptions)])
    return messages, feedback_text


def imagescoring(test_url, test_noise, few_shot_messages, feedback):
    prompt = (
        f"You are now shown a new CT scan. Noise: {test_noise:.3f}.\n"
        f"Below are previous prediction errors from similar cases:\n{feedback}\n\n"
        "Based on the examples above, evaluate its **technical image quality \n\n"
        "Use the same tone, criteria, and evaluation style as in the few-shot examples.\n"
        "Pay **particular attention to the provided noise level**\n"
        "If the noise level is low,the quality will be low and you will tend to give lower scores (for example, if the noise level is 0.008, this means high noise and low quality;\n"
        "if the noise level is 0.001, this means low noise and high quality and you will tend to give higher scores).\n"
        "However, balance it by taking into account all criteria, not just noise. So, you don't have to give a high score like 3.6 just because the noise is 0.001. What matters is quality.\n"
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
    outfile.write("RL-O3 Results\n")
    retry_list = []

    for i, (filename, info) in enumerate(test_data.items()):
        url = info["url"]
        region = info["region"]
        noise_level = test_noise.get(filename.replace(".png", ".tif"), 0.0)

        few_shots, feedback = fewshotmessages_with_prediction_feedback(
            train_data, train_scores, train_noise, region, batch_size=5
        )

        print(f"\n[{i+1}/{len(test_data)}] {filename} | Region: {region} | Noise: {noise_level:.3f}")
        result = imagescoring(url, noise_level, few_shots, feedback)
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
        for attempt in range(5):  # up to 5 attempts
            still_failed = []
            for filename, url, region, noise_level in retry_list:
                few_shots, feedback = fewshotmessages_with_prediction_feedback(
                    train_data, train_scores, train_noise, region, batch_size=5
                )

                print(f"Retrying {filename} (attempt {attempt+1})...")
                result = imagescoring(url, noise_level, few_shots, feedback)
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
        print(f"\nFinal failed cases: {len(retry_list)} — Could not be parsed after retries.")
        with open(output_path, 'a', encoding='utf-8') as outfile:
            for filename, _, _, _ in retry_list:
                outfile.write(f'"{filename.replace(".png", ".tif")}": ERROR - final retry failed\n')
