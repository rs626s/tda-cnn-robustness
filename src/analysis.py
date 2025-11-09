from typing import Dict, List
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import json
import os


# -------------------------------------------------------------------------
# Tests model accuracy under different Gaussian noise levels.
# Inputs: model, testLoader, device, noiseLevels, maxBatches (optional)
# Output: dict {noise_sigma: accuracy}
# -------------------------------------------------------------------------
def testNoiseRobustness(model, testLoader, device, noiseLevels=[0.0, 0.1, 0.2, 0.3], maxBatches=None):
    results = {}
    model.eval()
    
    for sigma in noiseLevels:
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(testLoader):
            if maxBatches is not None and batch_idx >= maxBatches:
                break

            images, labels = images.to(device), labels.to(device)
            
            if sigma > 0:
                noise = torch.randn_like(images) * sigma
                noisyImages = torch.clamp(images + noise, 0, 1)
            else:
                noisyImages = images
            
            with torch.no_grad():
                outputs = model(noisyImages)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        results[f'noise_sigma_{sigma}'] = correct / total
    
    return results


# -------------------------------------------------------------------------
# Plot robustness curves for different noise levels
# Inputs: robustnessResults, savePath, modelName, datasetName
# Output: saved robustness plot (.png)
# -------------------------------------------------------------------------
def plotRobustnessCurves(robustnessResults, savePath, modelName, datasetName):
    sigmas = [k for k in robustnessResults.keys() if k.startswith('noise_sigma_')]
    accuracies = [robustnessResults[s] for s in sigmas]
    sigmaValues = [float(s.split('_')[-1]) for s in sigmas]
    
    plt.figure(figsize=(8, 5))
    plt.plot(sigmaValues, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Noise Sigma')
    plt.ylabel('Accuracy')
    plt.title(f'Robustness: {modelName} on {datasetName}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------------------
# Extract and visualize topological features from the PLLay layer.
# Inputs: model, testLoader, device, savePath, maxBatches (optional)
# Output: scatter plot of topological feature space (.png)
# -------------------------------------------------------------------------
def analyzeTopologicalFeatures(model, testLoader, device, savePath, maxBatches=None):
    model.eval()
    topologyFeatures = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testLoader):
            if maxBatches is not None and batch_idx >= maxBatches:
                break
            images = images.to(device)
            
            # Extract topological features directly from PLLay layer
            if hasattr(model, 'topologyLayer'):
                topoFeats = model.topologyLayer(images)
                topologyFeatures.append(topoFeats.cpu())
                labels.append(targets)
    
    if topologyFeatures:
        topologyFeatures = torch.cat(topologyFeatures).numpy()
        labels = torch.cat(labels).numpy()
        
        # Create topological feature visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(topologyFeatures[:, 0], topologyFeatures[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar()
        plt.title('Topological Feature Space (First 2 Dimensions)')
        plt.xlabel('Topological Feature 1')
        plt.ylabel('Topological Feature 2')
        plt.tight_layout()
        plt.savefig(savePath)
        plt.close()
        
        print(f"Topological analysis saved to: {savePath}")
    else:
        print("No topological features found in model")


# -------------------------------------------------------------------------
# Collect and aggregate metrics from all experiment folders.
# Inputs: resultsDir (default: "outputs")
# Output: pandas DataFrame containing all run metrics
# -------------------------------------------------------------------------
def collectResults(resultsDir: str = "outputs"):
    allResults = []    
    for runDir in os.listdir(resultsDir):
        metricsPath = os.path.join(resultsDir, runDir, "metrics.json")
        if os.path.exists(metricsPath):
            with open(metricsPath, 'r') as f:
                metrics = json.load(f)
                metrics['run_dir'] = runDir
                allResults.append(metrics)
    
    return pd.DataFrame(allResults)


# -------------------------------------------------------------------------
# Generate comparison plots of model accuracy across datasets.
# Inputs: df (results DataFrame), outputDir (path to save plots)
# Output: bar plot of model performance (.png)
# -------------------------------------------------------------------------
def createComparisonPlots(df: pd.DataFrame, outputDir: str):
    if len(df) == 0:
        print("No results to plot")
        return
    
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model', y='test_acc', hue='dataset')
    plt.title('Model Comparison Across Datasets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, 'model_comparison.png'))
    plt.close()