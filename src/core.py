from datetime import datetime
import torch
import os


# -------------------------------------------------------------------------
# Create a timestamped directory for the current experiment run.
# Inputs: baseOutputDir (root path), tag (run identifier)
# Output: full path of the newly created run directory
# -------------------------------------------------------------------------
def getRunDir(baseOutputDir: str, tag: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runDir = os.path.join(baseOutputDir, f"{tag}_{timestamp}")
    os.makedirs(runDir, exist_ok=True)
    return runDir


# -------------------------------------------------------------------------
# Set up experiment components: device, data loaders, model, and output dir.
# Inputs: args (CLI arguments), mode ('train' or 'eval')
# Output: dict with device, model, loaders, metadata, and run directory
# -------------------------------------------------------------------------
def setupExperiment(args, mode: str = "train") -> dict:    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                #Detect available device (GPU if available, otherwise CPU)
    
    #Import here to avoid circular imports
    import data
    import models
    import utils
    
    #Load appropriate data splits based on mode
    trainLoader, valLoader, testLoader, inCh, numClasses, imgSize = data.makeLoaders(
        datasetName=args.dataset,
        batchSize=args.batchSize if mode == "train" else 128,
        valSplit=args.valSplit if mode == "train" else 0.1,
        numWorkers=2,
        seed=args.seed if mode == "train" else 42,
        args=args
    )
    
    #Build the specified model architecture
    if args.model.startswith("vgg"):
        img_size = (32, 32)
    else:
        img_size = (28, 28)

    useGudhiFlag = getattr(args, "useGudhi", False)    
    model = models.buildModel(args.model, inCh, imgSize, numClasses, useGudhi=useGudhiFlag).to(device)
    
    #Create unique run directory for outputs
    tag = args.tag or f"{mode}_{args.model}_{args.dataset}"
    outputDir = args.outDir if mode == "train" else getattr(args, 'outDir', 'outputs_eval')
    runDir = getRunDir(outputDir, tag)
    
    return {
        'device': device,
        'model': model,
        'trainLoader': trainLoader,
        'valLoader': valLoader,
        'testLoader': testLoader,
        'inCh': inCh,
        'numClasses': numClasses,
        'imgSize': imgSize,
        'runDir': runDir
    }


# -------------------------------------------------------------------------
# Load model weights from a saved checkpoint file (.pt).
# Inputs: model (architecture), checkpointPath (file path), device (CPU/GPU)
# Output: model with restored weights
# -------------------------------------------------------------------------
def loadModelFromCheckpoint(model: torch.nn.Module, checkpointPath: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(checkpointPath):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpointPath}")
        
    checkpoint = torch.load(checkpointPath, map_location=device)                    #Load checkpoint and restore model weights
    model.load_state_dict(checkpoint["model_state"])
    return model


# -------------------------------------------------------------------------
# Save evaluation metrics and confusion matrix to the run directory.
# Inputs: evalStats (dict), model, args, runDir (output path), classNames (labels)
# Output: metrics files (metrics.json / metrics.pt) and confusion_matrix.png
# -------------------------------------------------------------------------
def saveEvaluationArtifacts(evalStats: dict, model: torch.nn.Module, args, runDir: str, classNames: list):
    import utils
    import json

    metrics = {
        "test_acc": float(evalStats["acc"]),
        "params": int(utils.countParameters(model)),
        "model": args.model,
        "dataset": args.dataset,
        "eval_noise": getattr(args, 'evalNoise', 'none'),
        "noise_sigma": getattr(args, 'noiseSigma', 0.0),
        "checkpoint": getattr(args, 'checkpoint', 'N/A'),
    }

    torch.save(metrics, os.path.join(runDir, "metrics.pt"))
    with open(os.path.join(runDir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    utils.saveConfusionMatrix(
        yTrue=evalStats["targets"].numpy(),
        yPred=evalStats["preds"].numpy(),
        classes=classNames,
        path=os.path.join(runDir, "confusion_matrix.png")
    )