from torch.nn.parameter import UninitializedParameter
from sklearn.metrics import confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import os


# -------------------------------------------------------------------------
# Save model checkpoint with training state.
# Inputs: model, optimizer, epoch (int), path (save location)
# Output: checkpoint file (.pt) containing model and optimizer states
# -------------------------------------------------------------------------
def saveCheckpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(state, path)


# -------------------------------------------------------------------------
# Load model and optimizer state from a checkpoint file.
# Inputs: model, optimizer, path (checkpoint file), device (CPU/GPU)
# Output: last saved epoch number (int)
# -------------------------------------------------------------------------
def loadCheckpoint(model, optimizer, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint '{path}' not found.")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("epoch", 0)


# -------------------------------------------------------------------------
# Log a message with a timestamp to console and file.
# Inputs: message (str), logFile (optional, default='train.log')
# Output: writes formatted log entry with timestamp to file and prints to console
# -------------------------------------------------------------------------
def logMessage(message, logFile="train.log"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fullMessage = f"[{timestamp}] {message}"    
    print(fullMessage)

    try:
        with open(logFile, "a", encoding="utf-8") as f:
            f.write(fullMessage + "\n")
    except UnicodeEncodeError:
        safe_message = fullMessage.encode('utf-8', errors='replace').decode('utf-8')
        with open(logFile, "a", encoding="utf-8") as f:
            f.write(safe_message + "\n")



# -------------------------------------------------------------------------
# Count the total number of trainable parameters in a model.
# Inputs: model (torch.nn.Module)
# Output: integer count of all initialized trainable parameters
# -------------------------------------------------------------------------
def countParameters(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, UninitializedParameter):
            continue
        total += p.numel()
    return total



# -------------------------------------------------------------------------
# Compute top-1 classification accuracy from model logits.
# Inputs: logits (predicted outputs), targets (true labels)
# Output: accuracy value as a float
# -------------------------------------------------------------------------
def accuracyFromLogits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()



# -------------------------------------------------------------------------
# Generate and save a confusion matrix visualization as a PNG image.
# Inputs: yTrue (true labels), yPred (predicted labels),
#         classes (list of class names), path (save location)
# Output: saved image file 'confusion_matrix.png'
# -------------------------------------------------------------------------
def saveConfusionMatrix(yTrue, yPred, classes, path):
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



# -------------------------------------------------------------------------
# Plot and save training vs validation accuracy curves across epochs.
# Inputs: history (dict with 'train_acc' and 'val_acc'), outPath (save location)
# Output: saved plot image 'learning_curves.png'
# -------------------------------------------------------------------------
def plotLearningCurves(history, outPath):
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