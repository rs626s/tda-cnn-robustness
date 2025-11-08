import os
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from torch.nn.parameter import UninitializedParameter


# ===== Original utils functions =====
def saveCheckpoint(model, optimizer, epoch, path):
    """Saves model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(state, path)

def loadCheckpoint(model, optimizer, path, device):
    """Loads model checkpoint"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint '{path}' not found.")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("epoch", 0)

def logMessage(message, logFile="train.log"):
    """Logs message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fullMessage = f"[{timestamp}] {message}"
    
    print(fullMessage)
    
    # Use UTF-8 encoding to handle Unicode characters properly
    try:
        with open(logFile, "a", encoding="utf-8") as f:
            f.write(fullMessage + "\n")
    except UnicodeEncodeError:
        # Fallback: replace problematic Unicode characters
        safe_message = fullMessage.encode('utf-8', errors='replace').decode('utf-8')
        with open(logFile, "a", encoding="utf-8") as f:
            f.write(safe_message + "\n")


# ===== Metrics functions (moved from metrics.py) =====
def countParameters(model: torch.nn.Module) -> int:
    """Counts trainable parameters in a model, skipping uninitialized lazy params"""
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        # Skip lazy / uninitialized parameters (e.g., from nn.LazyLinear)
        if isinstance(p, UninitializedParameter):
            continue
        total += p.numel()
    return total

def accuracyFromLogits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes top-1 accuracy from raw logits"""
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def saveConfusionMatrix(yTrue, yPred, classes, path):
    """Saves confusion matrix as PNG"""
    cm = confusion_matrix(yTrue, yPred)
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tickMarks = np.arange(len(classes))
    plt.xticks(tickMarks, classes, rotation=45)
    plt.yticks(tickMarks, classes)
    
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

def plotLearningCurves(history, outPath):
    """Plots training and validation accuracy curves"""
    trainAcc = history.get("train_acc", [])
    valAcc = history.get("val_acc", [])
    
    fig = plt.figure(figsize=(6, 4))
    epochs = np.arange(1, len(trainAcc) + 1, dtype=int)
    
    plt.plot(epochs, trainAcc, label="train_acc")
    plt.plot(epochs, valAcc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    fig.savefig(outPath, bbox_inches='tight')
    plt.close(fig)