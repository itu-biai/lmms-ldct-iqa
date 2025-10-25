# -*- coding: utf-8 -*-
"""
Created on Thu May 29 17:22:24 2025

@author: kagan
"""

import os
import re
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

folderpath = "yourpath"
radiologscoresfile = os.path.join(folderpath, "Radiolog Scores Zero Shot Testing.txt")
modelfiles = [
    "GPT 4o Scores Zero Shot.txt",
    "Gemini Scores Zero Shot.txt",
    "O3 Scores Zero Shot.txt",
    "Gemma3 Scores Zero Shot.txt",
    "Grok2 Scores Zero Shot.txt",
    "Qwen Scores Zero Shot.txt",
    "GPT 4o-mini Scores Zero Shot.txt",
    "Meta Llama4 Scores Zero Shot.txt",
    "Claude Sonnet4 Scores Zero Shot.txt"
]


def modelscores(filepath):
    scores = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'"(.+?\.tif)":\s*([\d.]+)', line)
            if match:
                filename = match.group(1)
                score = float(match.group(2))
                scores[filename] = score
    return scores


radiologscores = modelscores(radiologscoresfile)
results = []


for model_file in modelfiles:
    model_path = os.path.join(folderpath, model_file)
    model_scores = modelscores(model_path)

    common_keys = sorted(set(model_scores.keys()) & set(radiologscores.keys()))
    if not common_keys:
        continue

    y_true = [radiologscores[k] for k in common_keys]
    y_pred = [model_scores[k] for k in common_keys]


    plcc, _ = pearsonr(y_true, y_pred)
    srocc, _ = spearmanr(y_true, y_pred)
    krocc, _ = kendalltau(y_true, y_pred)
    overall = abs(plcc) + abs(srocc) + abs(krocc)

    results.append({
        "Model": model_file.replace(".txt", ""),
        "PLCC": round(plcc, 4),
        "SROCC": round(srocc, 4),
        "KROCC": round(krocc, 4),
        "Overall Score": round(overall, 4)
    })


df = pd.DataFrame(results)
df = df.sort_values(by="Overall Score", ascending=False)
print(df)
