import torch
import torch.nn as nn
from tqdm import tqdm
from utils import accuracyFromLogits


def trainOneEpoch(model, loader, optimizer, device):
    """Trains the model for exactly one epoch"""
    model.train()
    lossFn = nn.CrossEntropyLoss()

    totalLoss, totalAcc, nSamples = 0.0, 0.0, 0

    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = lossFn(logits, targets)

        loss.backward()
        optimizer.step()

        batchSize = targets.size(0)
        totalLoss += loss.item() * batchSize
        totalAcc += (logits.argmax(1) == targets).float().sum().item()
        nSamples += batchSize

    return {
        "loss": totalLoss / max(nSamples, 1),
        "acc": totalAcc / max(nSamples, 1),
    }


@torch.no_grad()
def evaluateModel(model, loader, device, noiseFn=None, maxBatches=None):
    """Evaluates the model without gradient tracking"""
    model.eval()
    lossFn = nn.CrossEntropyLoss()

    totalLoss, totalAcc, nSamples = 0.0, 0.0, 0
    allPreds, allTargets = [], []

    # If maxBatches is given, tell tqdm the smaller total
    total_batches = maxBatches if maxBatches is not None else None

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="eval", leave=False, total=total_batches)):
        if maxBatches is not None and batch_idx >= maxBatches:
            break

        if noiseFn is not None:
            inputs = noiseFn(inputs)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = lossFn(logits, targets)

        batchSize = targets.size(0)
        totalLoss += loss.item() * batchSize
        batchAcc = accuracyFromLogits(logits, targets)
        totalAcc += batchAcc * batchSize
        nSamples += batchSize

        allPreds.append(torch.argmax(logits, dim=1).cpu())
        allTargets.append(targets.cpu())

    predsTensor = torch.cat(allPreds) if allPreds else torch.empty(0, dtype=torch.long)
    targetsTensor = torch.cat(allTargets) if allTargets else torch.empty(0, dtype=torch.long)

    return {
        "loss": totalLoss / max(nSamples, 1),
        "acc": totalAcc / max(nSamples, 1),
        "preds": predsTensor,
        "targets": targetsTensor,
    }