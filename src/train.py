# src/main.py
"""
Main training script for image classification models.
Handles the complete training pipeline from data loading to model evaluation.
"""

import argparse
import os
import torch
import torch.optim as optim

# Import local modules
import data
import trainers
import utils
import models
import analysis
import core

import threading
import time

class TimeoutThread(threading.Thread):
    def __init__(self, func, args=(), kwargs={}):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
    
    def run(self):
        try:
            self.result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.error = e

def run_with_timeout(func, timeout, args=(), kwargs={}):
    """Run a function with timeout on Windows"""
    thread = TimeoutThread(func, args, kwargs)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutError(f"Function exceeded {timeout} second timeout")
    
    if thread.error:
        raise thread.error
    
    return thread.result

def parseArguments() -> argparse.Namespace:
    """
    Parses command-line arguments to configure the training experiment.
    
    This function defines all the available options for training, including
    model architecture, dataset, hyperparameters, and evaluation settings.
    It provides a user-friendly interface to configure experiments without
    modifying code.
    
    Returns:
        argparse.Namespace: Object containing all parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train image classifiers with a unified interface.")
    
    # Model and dataset selection
    parser.add_argument("--model", required=True,
                       choices=["cnn", "cnn_topology", "resnet18", "resnet18_topology", "vgg11", "vgg11_topology"],
                       help="Model architecture to train")
    parser.add_argument("--dataset", required=True,
                       choices=["mnist", "fashion", "cifar10"],
                       help="Dataset to use for training")
    
    parser.add_argument("--useGudhi",
                        action="store_true",
                        help="Enable Gudhi-based persistence layer for topology models (default: use fast approximation)")
    parser.add_argument("--subsetSize",
                        type=int,
                        default=0,
                        help="If >0, limit the training set to this many samples (useful with --useGudhi).")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--batchSize", type=int, default=128, 
                       help="Mini-batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate for AdamW optimizer")
    parser.add_argument("--weightDecay", type=float, default=1e-4, 
                       help="L2 weight decay for regularization")
    
    # Data splitting and reproducibility
    parser.add_argument("--valSplit", type=float, default=0.1, 
                       help="Fraction of training data to use for validation")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducible results")
    
    # Robustness evaluation options
    parser.add_argument("--evalNoise", choices=["none", "gaussian"], default="none",
                       help="Type of noise to apply during final test evaluation")
    parser.add_argument("--noiseSigma", type=float, default=0.0, 
                       help="Standard deviation for Gaussian noise during evaluation")
    
    # Systematic robustness testing
    parser.add_argument("--testRobustness", action="store_true", 
                       help="Enable systematic robustness testing with multiple noise levels")
    parser.add_argument("--robustnessLevels", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3],
                       help="Noise levels to test for robustness evaluation")
    parser.add_argument("--robustnessMaxBatches", type=int, default=None,
                        help="Limit number of test batches for robustness evaluation (for quick experiments)")
    
    # Output configuration
    parser.add_argument("--outDir", type=str, default="outputs", 
                       help="Directory to store all training outputs")
    parser.add_argument("--tag", type=str, default=None, 
                       help="Custom tag for the experiment run directory")
    
    return parser.parse_args()


def main():
    """
    Main training pipeline execution function.
    
    This function orchestrates the entire training process:
    1. Parse command-line arguments
    2. Setup experiment environment
    3. Train model for specified epochs
    4. Evaluate on test set
    5. Run robustness analysis (if requested)
    6. Save all artifacts and results
    """
    # Parse command-line arguments to get experiment configuration
    args = parseArguments()
    
    # Setup experiment components using shared core functionality
    setup = core.setupExperiment(args, mode="train") # type: ignore
    
    # Initialize optimizer for model training
    optimizer = optim.AdamW(setup['model'].parameters(), lr=args.lr, weight_decay=args.weightDecay) # type: ignore
    
    # Setup logging and output directory
    logPath = os.path.join(setup['runDir'], "train.log")
    utils.logMessage(f"Run directory: {setup['runDir']}", logPath)
    utils.logMessage(f"Experiment arguments: {vars(args)}", logPath)
    utils.logMessage(f"Model parameters: {utils.countParameters(setup['model']):,}", logPath)
    
    # Training loop tracking variables
    trainingHistory = {"train_acc": [], "val_acc": []}
    bestValidationAccuracy = 0.0
    bestCheckpointPath = os.path.join(setup['runDir'], "model_best.pt")
    useTimeout = args.model not in ["cnn_topology", "resnet18_topology", "vgg11_topology"]

    # Main training loop over epochs
    for epoch in range(1, args.epochs + 1):
        try:
            if useTimeout:
                trainStats = run_with_timeout(
                    trainers.trainOneEpoch,
                    timeout=300,
                    args=(setup['model'], setup['trainLoader'], optimizer, setup['device']),
                )
            else:
                # Topology models may be slower; run without timeout
                trainStats = trainers.trainOneEpoch(
                    setup['model'], setup['trainLoader'], optimizer, setup['device']
                )
        except TimeoutError:
            utils.logMessage(f"Epoch {epoch} training timed out after 5 minutes", logPath)
            break
        except Exception as e:
            utils.logMessage(f"Error in epoch {epoch}: {e}", logPath)
            break
        # Validation after training epoch
        validationStats = trainers.evaluateModel(setup['model'], setup['valLoader'], setup['device'])
        
        # Record training progress
        trainingHistory["train_acc"].append(trainStats["acc"])
        trainingHistory["val_acc"].append(validationStats["acc"])
        
        # Log epoch results
        utils.logMessage(
            f"Epoch {epoch:03d} | "
            f"train_loss={trainStats['loss']:.4f} train_acc={trainStats['acc']:.4f} | "
            f"val_loss={validationStats['loss']:.4f} val_acc={validationStats['acc']:.4f}",
            logPath
        )
        
        # Save checkpoint if this is the best model so far
        if validationStats["acc"] > bestValidationAccuracy:
            bestValidationAccuracy = validationStats["acc"]
            utils.saveCheckpoint(setup['model'], optimizer, epoch, bestCheckpointPath)
            utils.logMessage(f"  -> Saved best checkpoint (val_acc={bestValidationAccuracy:.4f})", logPath)
    
    # Load best model for final evaluation
    if os.path.exists(bestCheckpointPath):
        bestModel = core.loadModelFromCheckpoint(setup['model'], bestCheckpointPath, setup['device'])
    else:
        utils.logMessage(
            f"No best checkpoint found at {bestCheckpointPath}. "
            "Training may have stopped early; using last model weights instead.",
            logPath,
        )
        bestModel = setup['model']
    
    # Standard test evaluation
    noiseFunction = None
    if args.evalNoise != "none":
        noiseFunction = (lambda x: data.applyEvalNoise(x, noiseType=args.evalNoise, sigma=args.noiseSigma))
    
    testStats = trainers.evaluateModel(bestModel, setup['testLoader'], setup['device'], noiseFn=noiseFunction, maxBatches=args.robustnessMaxBatches)
    
    # Systematic robustness testing (if requested)
    robustnessResults = {}
    if args.testRobustness:
        utils.logMessage("Starting systematic robustness testing...", logPath)
        
        # Test model performance across multiple noise levels
        robustnessResults = analysis.testNoiseRobustness(
            bestModel, setup['testLoader'], setup['device'], noiseLevels=args.robustnessLevels, maxBatches=args.robustnessMaxBatches
        )
        
        # Log robustness results
        for noiseLevel, accuracy in robustnessResults.items():
            utils.logMessage(f"Robustness {noiseLevel}: {accuracy:.4f}", logPath)
        
        # Create robustness visualization
        analysis.plotRobustnessCurves(
            robustnessResults, 
            os.path.join(setup['runDir'], "robustness_curves.png"),
            modelName=args.model,
            datasetName=args.dataset
        )
    
    # Save training artifacts and results
    metricsDict = {
        "best_val_acc": float(bestValidationAccuracy),
        "test_acc": float(testStats["acc"]),
        "params": int(utils.countParameters(bestModel)),
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "eval_noise": args.evalNoise,
        "noise_sigma": args.noiseSigma,
    }
    
    # Include robustness results in metrics
    metricsDict.update(robustnessResults)
    
    # Save metrics in multiple formats
    torch.save(metricsDict, os.path.join(setup['runDir'], "metrics.pt"))
    import json
    with open(os.path.join(setup['runDir'], "metrics.json"), "w") as f:
        json.dump(metricsDict, f, indent=2)
    
    # Save visualizations
    classNames = [str(i) for i in range(setup['numClasses'])]
    utils.saveConfusionMatrix(
        yTrue=testStats["targets"].numpy(),
        yPred=testStats["preds"].numpy(),
        classes=classNames,
        path=os.path.join(setup['runDir'], "confusion_matrix.png")
    )
    
    utils.plotLearningCurves(trainingHistory, os.path.join(setup['runDir'], "learning_curves.png"))
    
    # Special analysis for topology-enhanced models
    if "topology" in args.model:
        utils.logMessage("Running topological feature analysis...", logPath)

        # --- Limit analysis batches using the same CLI flag ---
        subsetSize = args.robustnessMaxBatches * args.batchSize if args.robustnessMaxBatches else len(setup['testLoader'].dataset)
        subset_indices = list(range(0, min(subsetSize, len(setup['testLoader'].dataset))))

        subset = torch.utils.data.Subset(setup['testLoader'].dataset, subset_indices)
        subsetLoader = torch.utils.data.DataLoader(subset, batch_size=args.batchSize, shuffle=False)

        # Run analysis only on this subset, with optional maxBatches inside the function
        analysis.analyzeTopologicalFeatures(
            bestModel,
            subsetLoader,
            setup['device'],
            savePath=os.path.join(setup['runDir'], "topology_analysis.png"),
            maxBatches=args.robustnessMaxBatches
        )

    
    # Final summary
    utils.logMessage(f"Training complete. Best val_acc={bestValidationAccuracy:.4f} | test_acc={testStats['acc']:.4f}", logPath)
    
    # Robustness summary (if tested)
    if robustnessResults:
        cleanAccuracy = robustnessResults.get('noise_sigma_0.0', testStats["acc"])
        robustAccuracy01 = robustnessResults.get('noise_sigma_0.1', 0)
        robustAccuracy02 = robustnessResults.get('noise_sigma_0.2', 0)
        utils.logMessage(f"Robustness Summary: Clean={cleanAccuracy:.4f}, sigma=0.1={robustAccuracy01:.4f}, sigma=0.2={robustAccuracy02:.4f}", logPath)
    
    print("All artifacts saved to:", setup['runDir'])


if __name__ == "__main__":
    main()