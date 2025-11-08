from typing import Tuple, Callable, Optional
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torchvision.datasets as dsets


def meanStd(datasetName: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Returns channel-wise (mean, std) for normalization.
    """
    if datasetName in ["mnist", "fashion"]:
        return (0.1307,), (0.3081,)
    elif datasetName in ["cifar10", "cifar100"]:
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        raise ValueError(f"Unknown dataset '{datasetName}'")


def buildTransforms(datasetName: str, isTrain: bool) -> T.Compose:
    """
    Builds torchvision transforms for a dataset.
    """
    mean, std = meanStd(datasetName)

    if datasetName in ["mnist", "fashion"]:
        aug = []
        if isTrain:
            aug += [T.RandomRotation(10)]
        aug += [T.Resize((32, 32)), T.ToTensor(), T.Normalize(mean, std)]
        return T.Compose(aug)

    elif datasetName in ["cifar10", "cifar100"]:
        aug = []
        if isTrain:
            aug += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        aug += [T.Resize((32, 32)), T.ToTensor(), T.Normalize(mean, std)]
        return T.Compose(aug)

    else:
        raise ValueError(f"Unknown dataset '{datasetName}'")


def loadDataset(datasetName: str, isTrain: bool, root: str = "./data"):
    """
    Loads a torchvision dataset with the appropriate transforms.
    """
    if datasetName == "mnist":
        return dsets.MNIST(root, train=isTrain, download=True, transform=buildTransforms("mnist", isTrain))
    if datasetName == "fashion":
        return dsets.FashionMNIST(root, train=isTrain, download=True, transform=buildTransforms("fashion", isTrain))
    if datasetName == "cifar10":
        return dsets.CIFAR10(root, train=isTrain, download=True, transform=buildTransforms("cifar10", isTrain))
    raise ValueError("Dataset not supported. Use one of: mnist, fashion, cifar10.")


def makeLoaders(
    datasetName: str,
    batchSize: int = 128,
    valSplit: float = 0.1,
    numWorkers: int = 2,
    seed: int = 42,
    args=None
):
    """
    Creates train/val/test DataLoaders and returns basic dataset metadata.
    """
    # Load full training split
    fullTrain = loadDataset(datasetName, isTrain=True)
    # Load official test split
    testSet = loadDataset(datasetName, isTrain=False)

    if args is not None and getattr(args, "useGudhi", False):
        subsetSize = getattr(args, "subsetSize", 5000)
        subsetSize = min(subsetSize, len(fullTrain))
        fullTrain = torch.utils.data.Subset(fullTrain, range(subsetSize))

    # Derive validation length
    valLen = int(len(fullTrain) * valSplit)
    trainLen = len(fullTrain) - valLen

    # Reproducible split using a fixed seed
    generator = torch.Generator().manual_seed(seed)
    trainSet, valSet = random_split(fullTrain, [trainLen, valLen], generator=generator)

    # Only use pin_memory if CUDA is available (FIXED)
    pin_memory = torch.cuda.is_available()
    
    # DataLoaders with conditional pin_memory
    trainLoader = DataLoader(
        trainSet, 
        batch_size=batchSize, 
        shuffle=True, 
        num_workers=numWorkers, 
        pin_memory=pin_memory
    )
    valLoader = DataLoader(
        valSet,   
        batch_size=batchSize, 
        shuffle=False, 
        num_workers=numWorkers, 
        pin_memory=pin_memory
    )
    testLoader = DataLoader(
        testSet,  
        batch_size=batchSize, 
        shuffle=False, 
        num_workers=numWorkers, 
        pin_memory=pin_memory
    )

    # Infer input channels and image size from dataset type
    inCh = 1 if datasetName in ["mnist", "fashion"] else 3
    numClasses = 10  # fixed for MNIST/Fashion/CIFAR-10
    imgSize = (32, 32)

    return trainLoader, valLoader, testLoader, inCh, numClasses, imgSize


def applyEvalNoise(x: torch.Tensor, noiseType: str = "none", sigma: float = 0.0) -> torch.Tensor:
    """
    Optionally corrupts a batch of inputs with noise during evaluation.
    """
    if noiseType == "none" or sigma <= 0.0:
        return x

    if noiseType == "gaussian":
        noise = torch.randn_like(x) * sigma
        xNoisy = torch.clamp(x + noise, -3.0, 3.0)
        return xNoisy

    raise ValueError(f"Unknown noiseType '{noiseType}' (use 'none' or 'gaussian').")