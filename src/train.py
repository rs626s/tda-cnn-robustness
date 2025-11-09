import torch.optim as optim
import threading
import trainers
import analysis
import argparse
import models
import torch
import utils
import data
import core
import time
import os


# -------------------------------------------------------------------------
# TimeoutThread: Executes a function in a separate thread with error capture.
# Inputs: func (callable), args (tuple), kwargs (dict)
# Output: stores result or error from the executed function
# -------------------------------------------------------------------------
class TimeoutThread(threading.Thread):
    def __init__(self, func, args=(), kwargs={}):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
    
    # ---------------------------------------------------------------------
    # Run the target function with provided arguments inside a thread.
    # Inputs: self.func, self.args, self.kwargs
    # Output: stores function output in self.result or exception in self.error
    # ---------------------------------------------------------------------
    def run(self):
        try:
            self.result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.error = e


# -------------------------------------------------------------------------
# Run a function with a specified timeout (Windows-compatible).
# Inputs: func (callable), timeout (seconds), args (tuple), kwargs (dict)
# Output: function result or raises TimeoutError/Exception if failed
# -------------------------------------------------------------------------
def run_with_timeout(func, timeout, args=(), kwargs={}):
    thread = TimeoutThread(func, args, kwargs)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutError(f"Function exceeded {timeout} second timeout")
    
    if thread.error:
        raise thread.error
    
    return thread.result


# -------------------------------------------------------------------------
# Parse command-line arguments for configuring the training experiment.
# Inputs: none (reads CLI arguments)
# Output: argparse.Namespace with model, dataset, hyperparameters, and settings
# -------------------------------------------------------------------------
def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classifiers with a unified interface.")
    
    #Model and dataset selection
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
    
    #Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--batchSize", type=int, default=128, 
                       help="Mini-batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate for AdamW optimizer")
    parser.add_argument("--weightDecay", type=float, default=1e-4, 
                       help="L2 weight decay for regularization")
    
    #Data splitting and reproducibility
    parser.add_argument("--valSplit", type=float, default=0.1, 
                       help="Fraction of training data to use for validation")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducible results")
    
    #Robustness evaluation options
    parser.add_argument("--evalNoise", choices=["none", "gaussian"], default="none",
                       help="Type of noise to apply during final test evaluation")
    parser.add_argument("--noiseSigma", type=float, default=0.0, 
                       help="Standard deviation for Gaussian noise during evaluation")
    
    #Systematic robustness testing
    parser.add_argument("--testRobustness", action="store_true", 
                       help="Enable systematic robustness testing with multiple noise levels")
    parser.add_argument("--robustnessLevels", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3],
                       help="Noise levels to test for robustness evaluation")
    parser.add_argument("--robustnessMaxBatches", type=int, default=None,
                        help="Limit number of test batches for robustness evaluation (for quick experiments)")
    
    #Output configuration
    parser.add_argument("--outDir", type=str, default="outputs", 
                       help="Directory to store all training outputs")
    parser.add_argument("--tag", type=str, default=None, 
                       help="Custom tag for the experiment run directory")
    
    return parser.parse_args()



# -------------------------------------------------------------------------
# Run the full training pipeline:
#  - parse CLI arguments and set up experiment
#  - train model with validation and checkpointing
#  - evaluate on test data with optional noise and robustness tests
#  - save metrics, plots, logs, and (optionally) topology analysis
# Inputs: none (uses parsed CLI arguments)
# Output: artifacts and metrics saved under the run directory
# -------------------------------------------------------------------------
def main():
    args = parseArguments()
    setup = core.setupExperiment(args, mode="train")

    optimizer = optim.AdamW(setup['model'].parameters(), lr=args.lr, weight_decay=args.weightDecay)
   
    logPath = os.path.join(setup['runDir'], "train.log")
    utils.logMessage(f"Run directory: {setup['runDir']}", logPath)
    utils.logMessage(f"Experiment arguments: {vars(args)}", logPath)
    utils.logMessage(f"Model parameters: {utils.countParameters(setup['model']):,}", logPath)
    

    trainingHistory = {"train_acc": [], "val_acc": []}
    bestValidationAccuracy = 0.0
    bestCheckpointPath = os.path.join(setup['runDir'], "model_best.pt")
    useTimeout = args.model not in ["cnn_topology", "resnet18_topology", "vgg11_topology"]

    for epoch in range(1, args.epochs + 1):
        try:
            if useTimeout:
                trainStats = run_with_timeout(
                    trainers.trainOneEpoch,
                    timeout=300,
                    args=(setup['model'], setup['trainLoader'], optimizer, setup['device']),
                )
            else:
                trainStats = trainers.trainOneEpoch(
                    setup['model'], setup['trainLoader'], optimizer, setup['device']
                )
        except TimeoutError:
            utils.logMessage(f"Epoch {epoch} training timed out after 5 minutes", logPath)
            break
        except Exception as e:
            utils.logMessage(f"Error in epoch {epoch}: {e}", logPath)
            break

        validationStats = trainers.evaluateModel(setup['model'], setup['valLoader'], setup['device'])       
        trainingHistory["train_acc"].append(trainStats["acc"])
        trainingHistory["val_acc"].append(validationStats["acc"])
        
        utils.logMessage(
            f"Epoch {epoch:03d} | "
            f"train_loss={trainStats['loss']:.4f} train_acc={trainStats['acc']:.4f} | "
            f"val_loss={validationStats['loss']:.4f} val_acc={validationStats['acc']:.4f}",
            logPath
        )
        
        if validationStats["acc"] > bestValidationAccuracy:
            bestValidationAccuracy = validationStats["acc"]
            utils.saveCheckpoint(setup['model'], optimizer, epoch, bestCheckpointPath)
            utils.logMessage(f"  -> Saved best checkpoint (val_acc={bestValidationAccuracy:.4f})", logPath)
    
    if os.path.exists(bestCheckpointPath):
        bestModel = core.loadModelFromCheckpoint(setup['model'], bestCheckpointPath, setup['device'])
    else:
        utils.logMessage(
            f"No best checkpoint found at {bestCheckpointPath}. "
            "Training may have stopped early; using last model weights instead.",
            logPath,
        )
        bestModel = setup['model']
    
    noiseFunction = None
    if args.evalNoise != "none":
        noiseFunction = (lambda x: data.applyEvalNoise(x, noiseType=args.evalNoise, sigma=args.noiseSigma))
    
    testStats = trainers.evaluateModel(bestModel, setup['testLoader'], setup['device'], noiseFn=noiseFunction, maxBatches=args.robustnessMaxBatches)
    
    robustnessResults = {}
    if args.testRobustness:
        utils.logMessage("Starting systematic robustness testing...", logPath)
        robustnessResults = analysis.testNoiseRobustness(
            bestModel, setup['testLoader'], setup['device'], noiseLevels=args.robustnessLevels, maxBatches=args.robustnessMaxBatches
        )

        for noiseLevel, accuracy in robustnessResults.items():
            utils.logMessage(f"Robustness {noiseLevel}: {accuracy:.4f}", logPath)

        analysis.plotRobustnessCurves(
            robustnessResults, 
            os.path.join(setup['runDir'], "robustness_curves.png"),
            modelName=args.model,
            datasetName=args.dataset
        )
    
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

    metricsDict.update(robustnessResults)
    torch.save(metricsDict, os.path.join(setup['runDir'], "metrics.pt"))
    
    import json
    with open(os.path.join(setup['runDir'], "metrics.json"), "w") as f:
        json.dump(metricsDict, f, indent=2)
    
    classNames = [str(i) for i in range(setup['numClasses'])]
    utils.saveConfusionMatrix(
        yTrue=testStats["targets"].numpy(),
        yPred=testStats["preds"].numpy(),
        classes=classNames,
        path=os.path.join(setup['runDir'], "confusion_matrix.png")
    )
    
    utils.plotLearningCurves(trainingHistory, os.path.join(setup['runDir'], "learning_curves.png"))
    
    if "topology" in args.model:
        utils.logMessage("Running topological feature analysis...", logPath)
        subsetSize = args.robustnessMaxBatches * args.batchSize if args.robustnessMaxBatches else len(setup['testLoader'].dataset)
        subset_indices = list(range(0, min(subsetSize, len(setup['testLoader'].dataset))))

        subset = torch.utils.data.Subset(setup['testLoader'].dataset, subset_indices)
        subsetLoader = torch.utils.data.DataLoader(subset, batch_size=args.batchSize, shuffle=False)
        analysis.analyzeTopologicalFeatures(
            bestModel,
            subsetLoader,
            setup['device'],
            savePath=os.path.join(setup['runDir'], "topology_analysis.png"),
            maxBatches=args.robustnessMaxBatches
        )

    utils.logMessage(f"Training complete. Best val_acc={bestValidationAccuracy:.4f} | test_acc={testStats['acc']:.4f}", logPath)
    if robustnessResults:
        cleanAccuracy = robustnessResults.get('noise_sigma_0.0', testStats["acc"])
        robustAccuracy01 = robustnessResults.get('noise_sigma_0.1', 0)
        robustAccuracy02 = robustnessResults.get('noise_sigma_0.2', 0)
        utils.logMessage(f"Robustness Summary: Clean={cleanAccuracy:.4f}, sigma=0.1={robustAccuracy01:.4f}, sigma=0.2={robustAccuracy02:.4f}", logPath)
    
    print("All artifacts saved to:", setup['runDir'])


if __name__ == "__main__":
    main()