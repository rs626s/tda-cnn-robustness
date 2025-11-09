import argparse
import os
import torch
import core
import trainers
import utils
import analysis
import data


# -------------------------------------------------------------------------
# Parse command-line arguments for model evaluation configuration.
# Inputs: none (reads CLI arguments)
# Output: argparse.Namespace with evaluation parameters
# -------------------------------------------------------------------------
def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    
    #Model and dataset specification
    parser.add_argument("--model", required=True,
                       choices=["cnn", "cnn_topology", "resnet18", "resnet18_topology", "vgg11", "vgg11_topology"],
                       help="Model architecture to evaluate")
    parser.add_argument("--dataset", required=True,
                       choices=["mnist", "fashion", "cifar10"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--checkpoint", required=True, type=str,
                       help="Path to the saved checkpoint file (.pt format)")
    
    parser.add_argument(
        "--useGudhi",
        action="store_true",
        help="Enable Gudhi-based persistence layer for topology models during evaluation",
    )
    
    #Evaluation noise options
    parser.add_argument("--evalNoise", choices=["none", "gaussian"], default="none",
                       help="Type of noise to apply during evaluation")
    parser.add_argument("--noiseSigma", type=float, default=0.0,
                       help="Standard deviation for Gaussian noise")
    
    #Systematic robustness testing
    parser.add_argument("--testRobustness", action="store_true", 
                       help="Enable systematic robustness testing with multiple noise levels")
    parser.add_argument("--robustnessLevels", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3],
                       help="Noise levels to test for robustness evaluation")
    
    #Output configuration
    parser.add_argument("--outDir", type=str, default="outputs_eval",
                       help="Directory to store evaluation outputs")
    parser.add_argument("--tag", type=str, default=None,
                       help="Custom tag for the evaluation run directory")
    
    return parser.parse_args()



# -------------------------------------------------------------------------
# Execute the full model evaluation pipeline:
#  - Load trained model and test dataset
#  - Apply noise and perform robustness analysis (optional)
#  - Generate and save evaluation results and plots
# Inputs: none (uses parsed CLI arguments)
# Output: evaluation metrics, plots, and logs saved to run directory
# -------------------------------------------------------------------------
def main():
    args = parseArguments()
    setup = core.setupExperiment(args, mode="eval")

    trainedModel = core.loadModelFromCheckpoint(setup['model'], args.checkpoint, setup['device'])

    logPath = os.path.join(setup['runDir'], "eval.log")
    utils.logMessage(f"Evaluation directory: {setup['runDir']}", logPath)
    utils.logMessage(f"Evaluation arguments: {vars(args)}", logPath)
    utils.logMessage(f"Model parameters: {utils.countParameters(trainedModel):,}", logPath)

    noiseFunction = None
    if args.evalNoise != "none":
        noiseFunction = (lambda x: data.applyEvalNoise(x, noiseType=args.evalNoise, sigma=args.noiseSigma))
    
    evaluationStats = trainers.evaluateModel(trainedModel, setup['testLoader'], setup['device'], noiseFn=noiseFunction)
    
    if args.testRobustness:
        utils.logMessage("Starting systematic robustness testing...", logPath)
        
        #Test model across multiple noise levels
        robustnessResults = analysis.testNoiseRobustness(
            trainedModel, setup['testLoader'], setup['device'], noiseLevels=args.robustnessLevels
        )
        
        #Log robustness performance at each noise level
        for noiseLevel, accuracy in robustnessResults.items():
            utils.logMessage(f"Robustness {noiseLevel}: {accuracy:.4f}", logPath)
        
        #Create robustness performance visualization
        analysis.plotRobustnessCurves(
            robustnessResults, 
            os.path.join(setup['runDir'], "robustness_curves.png"),
            modelName=args.model,
            datasetName=args.dataset
        )
    
    classNames = [str(i) for i in range(setup['numClasses'])]
    core.saveEvaluationArtifacts(evaluationStats, trainedModel, args, setup['runDir'], classNames)

    if "topology" in args.model:
        utils.logMessage("Running topological feature analysis...", logPath)
        analysis.analyzeTopologicalFeatures(
            trainedModel, setup['testLoader'], setup['device'],
            savePath=os.path.join(setup['runDir'], "topology_analysis.png")
        )

    utils.logMessage(f"Evaluation complete. Test accuracy: {evaluationStats['acc']:.4f}", logPath)
    print("Evaluation artifacts saved to:", setup['runDir'])


if __name__ == "__main__":
    main()