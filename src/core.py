import os
import torch
from datetime import datetime


def getRunDir(baseOutputDir: str, tag: str) -> str:
    """
    Creates a timestamped run directory for saving experiment artifacts.
    
    This function generates a unique directory name using the current timestamp
    and a user-provided tag. This ensures each experiment run has its own
    separate folder to avoid overwriting previous results.
    
    Args:
        baseOutputDir (str): Base directory where run folders will be created
        tag (str): Descriptive tag for this experiment run
        
    Returns:
        str: Full path to the newly created run directory
        
    Example:
        >>> getRunDir("outputs", "cnn_cifar10")
        'outputs/cnn_cifar10_2025-11-05_14-30-25'
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runDir = os.path.join(baseOutputDir, f"{tag}_{timestamp}")
    os.makedirs(runDir, exist_ok=True)
    return runDir


def setupExperiment(args, mode: str = "train") -> dict:
    """
    Sets up the common components needed for both training and evaluation.
    
    This function handles the repetitive setup tasks like device detection,
    data loading, model building, and directory creation. It returns a
    dictionary with all the setup components for easy access.
    
    Args:
        args: Command-line arguments containing experiment configuration
        mode (str): Either "train" or "eval" to specify the operation mode
        
    Returns:
        dict: Dictionary containing:
            - 'device': torch device (CPU/GPU)
            - 'model': Built and configured neural network model
            - 'trainLoader': Training data loader (None in eval mode)
            - 'valLoader': Validation data loader (None in eval mode) 
            - 'testLoader': Test data loader
            - 'inCh': Number of input channels (1 for grayscale, 3 for RGB)
            - 'numClasses': Number of output classes
            - 'imgSize': Tuple of (height, width) for input images
            - 'runDir': Path to the experiment output directory
    """
    # Detect available device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import here to avoid circular imports
    import data
    import models
    import utils
    
    # Load appropriate data splits based on mode
    trainLoader, valLoader, testLoader, inCh, numClasses, imgSize = data.makeLoaders(
        datasetName=args.dataset,
        batchSize=args.batchSize if mode == "train" else 128,
        valSplit=args.valSplit if mode == "train" else 0.1,
        numWorkers=2,
        seed=args.seed if mode == "train" else 42,
        args=args
    )
    
    # Build the specified model architecture
    if args.model.startswith("vgg"):
        img_size = (32, 32)
    else:
        img_size = (28, 28)

    useGudhiFlag = getattr(args, "useGudhi", False)    
    model = models.buildModel(args.model, inCh, imgSize, numClasses, useGudhi=useGudhiFlag).to(device)
    
    # Create unique run directory for outputs
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


def loadModelFromCheckpoint(model: torch.nn.Module, checkpointPath: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a trained model from a checkpoint file.
    
    This function handles the process of restoring model weights from a
    previously saved checkpoint. It's used during evaluation to load
    a pre-trained model for testing.
    
    Args:
        model (torch.nn.Module): The model architecture to load weights into
        checkpointPath (str): Path to the checkpoint file (.pt format)
        device (torch.device): Device to load the model onto
        
    Returns:
        torch.nn.Module: The model with loaded weights
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
    """
    if not os.path.isfile(checkpointPath):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpointPath}")
    
    # Load checkpoint and restore model weights
    checkpoint = torch.load(checkpointPath, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def saveEvaluationArtifacts(evalStats: dict, model: torch.nn.Module, args, runDir: str, classNames: list):
    """
    Saves evaluation results and visualizations to disk.
    
    This function creates the standard output files for an evaluation run,
    including metrics in both JSON and binary formats, and a confusion
    matrix visualization.
    
    Args:
        evalStats (dict): Dictionary containing evaluation statistics
        model (torch.nn.Module): The evaluated model
        args: Command-line arguments for metadata
        runDir (str): Directory to save the artifacts
        classNames (list): List of class names for confusion matrix
    """
    import utils
    import json
    
    # Compile comprehensive metrics dictionary
    metrics = {
        "test_acc": float(evalStats["acc"]),
        "params": int(utils.countParameters(model)),
        "model": args.model,
        "dataset": args.dataset,
        "eval_noise": getattr(args, 'evalNoise', 'none'),
        "noise_sigma": getattr(args, 'noiseSigma', 0.0),
        "checkpoint": getattr(args, 'checkpoint', 'N/A'),
    }
    
    # Save metrics in both binary (for programmatic use) and JSON (for human reading)
    torch.save(metrics, os.path.join(runDir, "metrics.pt"))
    with open(os.path.join(runDir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create and save confusion matrix visualization
    utils.saveConfusionMatrix(
        yTrue=evalStats["targets"].numpy(),
        yPred=evalStats["preds"].numpy(),
        classes=classNames,
        path=os.path.join(runDir, "confusion_matrix.png")
    )