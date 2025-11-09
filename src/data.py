from torch.utils.data import DataLoader, random_split
from typing import Tuple, Callable, Optional
import torchvision.transforms as T
import torchvision.datasets as dsets
import torch


# -------------------------------------------------------------------------
# Return channel-wise mean and standard deviation for dataset normalization.
# Inputs: datasetName ('mnist', 'fashion', 'cifar10', 'cifar100')
# Output: tuples (mean, std) for each image channel
# -------------------------------------------------------------------------
def meanStd(datasetName: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if datasetName in ["mnist", "fashion"]:
        return (0.1307,), (0.3081,)
    elif datasetName in ["cifar10", "cifar100"]:
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        raise ValueError(f"Unknown dataset '{datasetName}'")


# -------------------------------------------------------------------------
# Build data augmentation and normalization transforms for the dataset.
# Inputs: datasetName ('mnist', 'fashion', 'cifar10', 'cifar100'), isTrain (bool)
# Output: torchvision transform pipeline (T.Compose)
# -------------------------------------------------------------------------
def buildTransforms(datasetName: str, isTrain: bool) -> T.Compose:
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


# -------------------------------------------------------------------------
# Load the specified torchvision dataset with proper transforms applied.
# Inputs: datasetName ('mnist', 'fashion', 'cifar10'), isTrain (bool), root (data path)
# Output: torchvision dataset object
# -------------------------------------------------------------------------
def loadDataset(datasetName: str, isTrain: bool, root: str = "./data"):
    if datasetName == "mnist":
        return dsets.MNIST(root, train=isTrain, download=True, transform=buildTransforms("mnist", isTrain))
    if datasetName == "fashion":
        return dsets.FashionMNIST(root, train=isTrain, download=True, transform=buildTransforms("fashion", isTrain))
    if datasetName == "cifar10":
        return dsets.CIFAR10(root, train=isTrain, download=True, transform=buildTransforms("cifar10", isTrain))
    raise ValueError("Dataset not supported. Use one of: mnist, fashion, cifar10.")


# -------------------------------------------------------------------------
# Create train, validation, and test DataLoaders with optional Gudhi subset.
# Inputs: datasetName, batchSize, valSplit, numWorkers, seed, args (optional)
# Output: trainLoader, valLoader, testLoader, inCh, numClasses, imgSize
# -------------------------------------------------------------------------
def makeLoaders(datasetName: str, batchSize: int = 128, valSplit: float = 0.1, numWorkers: int = 2, seed: int = 42, args=None):

    fullTrain = loadDataset(datasetName, isTrain=True)
    testSet = loadDataset(datasetName, isTrain=False)

    if args is not None and getattr(args, "useGudhi", False):
        subsetSize = getattr(args, "subsetSize", 5000)
        subsetSize = min(subsetSize, len(fullTrain))
        fullTrain = torch.utils.data.Subset(fullTrain, range(subsetSize))

    valLen = int(len(fullTrain) * valSplit)
    trainLen = len(fullTrain) - valLen

    generator = torch.Generator().manual_seed(seed)
    trainSet, valSet = random_split(fullTrain, [trainLen, valLen], generator=generator)

    pin_memory = torch.cuda.is_available()
    
    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=pin_memory)
    valLoader   = DataLoader(valSet,   batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=pin_memory)
    testLoader  = DataLoader(testSet,  batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=pin_memory)

    inCh = 1 if datasetName in ["mnist", "fashion"] else 3
    numClasses = 10                                                             #fixed for MNIST/Fashion/CIFAR-10
    imgSize = (32, 32)

    return trainLoader, valLoader, testLoader, inCh, numClasses, imgSize


# -------------------------------------------------------------------------
# Apply Gaussian noise to input tensor during evaluation (optional).
# Inputs: x (tensor), noiseType ('none' or 'gaussian'), sigma (noise std)
# Output: noisy tensor (or original if noise disabled)
# -------------------------------------------------------------------------
def applyEvalNoise(x: torch.Tensor, noiseType: str = "none", sigma: float = 0.0) -> torch.Tensor:
    if noiseType == "none" or sigma <= 0.0:
        return x

    if noiseType == "gaussian":
        noise = torch.randn_like(x) * sigma
        xNoisy = torch.clamp(x + noise, -3.0, 3.0)
        return xNoisy

    raise ValueError(f"Unknown noiseType '{noiseType}' (use 'none' or 'gaussian').")