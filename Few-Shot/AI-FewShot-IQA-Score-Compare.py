# -*- coding: utf-8 -*-
"""
Created on Thu May 29 17:07:08 2025

@author: kagan
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

folderpath = "yourpath"
radiologscoresfile = os.path.join(folderpath, "Radiolog Scores Few Shot Testing.txt")

modelfiles = [
    "O3 Scores Few Shot.txt",
    "O3 Scores EF.txt",
    "O3-34 Scores Few Shot.txt",
    "O3-Label-Noise Scores Few Shot.txt"
    "yourfiles.txt"

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

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Line (y = x)')
    plt.xlabel("Radiologist Scores")
    plt.ylabel(f"{model_file.replace('.txt', '')} Scores")
    plt.title(f"Scatter: Radiologist vs {model_file.replace('.txt', '')}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(y_true, bins=10, alpha=0.6, label='Radiologist', edgecolor='black')
    plt.hist(y_pred, bins=10, alpha=0.6, label=model_file.replace(".txt", ""), edgecolor='black')
    plt.title(f"Histogram: Radiologist vs {model_file.replace('.txt', '')}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

df = pd.DataFrame(results)
df = df.sort_values(by="Overall Score", ascending=False)
print(df)
