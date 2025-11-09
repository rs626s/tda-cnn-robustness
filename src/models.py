from pllay import PersistenceLandscapeLayer
import torchvision.models as models
import torch.nn as nn
import torch
import gudhi


# -------------------------------------------------------------------------
# Build and return the specified neural network architecture.
# Inputs: modelName, inCh, imgSize, numClasses, useGudhi (bool for topology)
# Output: initialized model instance
# -------------------------------------------------------------------------
def buildModel(modelName, inCh, imgSize, numClasses, useGudhi: bool = False):
    modelName = modelName.lower()
    
    if modelName == "cnn":
        return SimpleCnn(inCh=inCh, numClasses=numClasses, imgSize=imgSize)
    if modelName == "cnn_topology":
        return CnnWithTopologyPreprocess(inCh=inCh, numClasses=numClasses, imgSize=imgSize, useGudhi=useGudhi)
    if modelName == "resnet18":
        return buildResnet("resnet18", numClasses=numClasses, inCh=inCh, weights=None)
    if modelName == "resnet18_topology":
        return ResNetWithTopologyPreprocess("resnet18", numClasses=numClasses, inCh=inCh, imgSize=imgSize, useGudhi=useGudhi)
    if modelName == "vgg11":
        return buildVgg("vgg11", numClasses=numClasses, inCh=inCh, weights=None)
    if modelName == "vgg11_topology":
        return VggWithTopologyPreprocess("vgg11", numClasses=numClasses, inCh=inCh, imgSize=imgSize, useGudhi=useGudhi)
    
    raise ValueError(f"Unknown model '{modelName}'")



# -------------------------------------------------------------------------
# Neural module for extracting and transforming topological features using PLLay.
# Inputs: outputDim (feature size), useGudhi (bool), maxPoints (for Gudhi)
# Output: tensor of topological feature vectors (B * outputDim)
# -------------------------------------------------------------------------
class TopologicalPreprocess(nn.Module):
    def __init__(self, outputDim=64, useGudhi: bool = False, maxPoints: int = 256):
        super().__init__()
        self.outputDim = outputDim

        self.plLayer = PersistenceLandscapeLayer(
            filtrationType="dtm",
            m0=0.05,
            kMax=3,
            resolution=outputDim,
            useGudhi=useGudhi,                                                  #fast learned descriptor
            maxPoints=maxPoints,
        )

        self.featureTransform = nn.Sequential(
            nn.Linear(outputDim, 128),
            nn.ReLU(),
            nn.Linear(128, outputDim),
        )

    def forward(self, x):
        try:
            topologicalRaw = self.plLayer(x)                                        #(B, outputDim)
            topologicalFeatures = self.featureTransform(topologicalRaw)
        except Exception as e:
            batchSize = x.shape[0]
            print(f"Topology computation failed, using zeros: {e}")
            topologicalFeatures = torch.zeros(batchSize, self.outputDim, device=x.device)

        return topologicalFeatures
    

# -------------------------------------------------------------------------
# CNN model augmented with a topological preprocessing channel (PLLay-based).
# Inputs: inCh, numClasses, imgSize, topologyDim, useGudhi (bool), maxPoints
# Output: class logits from CNN with integrated topological features
# -------------------------------------------------------------------------
class CnnWithTopologyPreprocess(nn.Module):
    def __init__(self, inCh, numClasses, imgSize, topologyDim=64, useGudhi: bool = False, maxPoints: int = 256):
        super().__init__()
        self.imgSize = imgSize
        self.topologyLayer = TopologicalPreprocess(outputDim=topologyDim, useGudhi=useGudhi, maxPoints=maxPoints)
        
        self.cnnInputChannels = inCh + 1                                        #CNN will see original image + 1 topology channel
        self.cnn = nn.Sequential(
            nn.Conv2d(self.cnnInputChannels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.cnnOutputDim = self.getCnnOutputDim()
        self.classifier = nn.Sequential(
            nn.Linear(self.cnnOutputDim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, numClasses),
        )

    def getCnnOutputDim(self):
        with torch.no_grad():
            dummyInput = torch.zeros(
                1,
                self.cnnInputChannels,
                self.imgSize[0],
                self.imgSize[1],
            )
            dummyOutput = self.cnn(dummyInput)
            return dummyOutput.shape[1]

    def buildTopologyChannel(self, x):
        topologyFeatures = self.topologyLayer(x)                                            #(batch, topologyDim)
        batchSize, _, h, w = x.shape

        firstFeature = topologyFeatures[:, 0].view(batchSize, 1, 1, 1)
        topologyChannel = firstFeature.expand(-1, 1, h, w)                                  #(batch, 1, H, W)
        return topologyChannel

    def forward(self, x):        
        topologyChannel = self.buildTopologyChannel(x)                                      #Step1: Pre-processing: persistence on the raw image        
        cnnInput = torch.cat([x, topologyChannel], dim=1)                                   #Step2: Concatenate persistence as extra input channel (pre-CNN)        
        cnnFeatures = self.cnn(cnnInput)                                                    #Step3: Standard CNN pipeline
        logits = self.classifier(cnnFeatures)
        return logits



# -------------------------------------------------------------------------
# ResNet model extended with a topological preprocessing channel (PLLay-based).
# Inputs: modelName, numClasses, inCh, imgSize, topologyDim, useGudhi, maxPoints
# Output: class logits from ResNet with integrated topological features
# -------------------------------------------------------------------------
class ResNetWithTopologyPreprocess(nn.Module):
    def __init__(self, modelName, numClasses, inCh=3, imgSize=(32, 32), topologyDim=64, useGudhi: bool = False, maxPoints: int = 256):
        super().__init__()
        self.imgSize = imgSize
        self.topologyLayer = TopologicalPreprocess(outputDim=topologyDim, useGudhi=useGudhi, maxPoints=maxPoints)        
        self.resnetInputChannels = inCh + 1                                     #ResNet will see original image + 1 topology channel        
        self.resnet = buildResnet(                                              #Build base ResNet that accepts extra channel
            modelName,
            numClasses=numClasses,
            inCh=self.resnetInputChannels,
            weights=None,
        )
        
        originalConv = self.resnet.conv1                                        #Modify first conv to match new input channels
        self.resnet.conv1 = nn.Conv2d(
            self.resnetInputChannels,
            originalConv.out_channels,
            kernel_size=originalConv.kernel_size,
            stride=originalConv.stride,
            padding=originalConv.padding,
            bias=originalConv.bias is not None,
        )

    def buildTopologyChannel(self, x):
        topologyFeatures = self.topologyLayer(x)                                #(batch, topologyDim)
        batchSize, _, h, w = x.shape

        topologyChannel = topologyFeatures.unsqueeze(-1).unsqueeze(-1)
        topologyChannel = topologyChannel.expand(-1, -1, h, w)
        topologyInput = topologyChannel[:, 0:1, :, :]                           #(batch, 1, H, W)
        return topologyInput

    def forward(self, x):        
        topologyInput = self.buildTopologyChannel(x)                            #Step1: Pre-processing: persistence on raw image        
        resnetInput = torch.cat([x, topologyInput], dim=1)                      #Step2: Concatenate as input to ResNet (pre-CNN)        
        return self.resnet(resnetInput)                                         #Step3: Standard ResNet forward



# -------------------------------------------------------------------------
# VGG model extended with a topological preprocessing channel (PLLay-based).
# Inputs: modelName, numClasses, inCh, imgSize, topologyDim, useGudhi, maxPoints
# Output: class logits from VGG with integrated topological features
# -------------------------------------------------------------------------
class VggWithTopologyPreprocess(nn.Module):
    def __init__(self, modelName, numClasses, inCh=3, imgSize=(32, 32), topologyDim=64, useGudhi: bool = False, maxPoints: int = 256):
        super().__init__()
        self.imgSize = imgSize
        self.topologyLayer = TopologicalPreprocess(outputDim=topologyDim, useGudhi=useGudhi, maxPoints=maxPoints)      
        self.vggInputChannels = inCh + 1                                        #VGG will see original image + 1 topology channel        
        self.vgg = buildVgg(                                                    #Build base VGG that accepts extra channel    
            modelName,
            numClasses=numClasses,
            inCh=self.vggInputChannels,
            weights=None,
        )
        
        originalConv = self.vgg.features[0]                                     #Modify first conv to accept topology channel
        self.vgg.features[0] = nn.Conv2d(
            self.vggInputChannels,
            originalConv.out_channels,
            kernel_size=originalConv.kernel_size,
            stride=originalConv.stride,
            padding=originalConv.padding,
            bias=originalConv.bias is not None,
        )

    def buildTopologyChannel(self, x):
        topologyFeatures = self.topologyLayer(x)                                #(batch, topologyDim)
        batchSize, _, h, w = x.shape

        topologyChannel = topologyFeatures.unsqueeze(-1).unsqueeze(-1)
        topologyChannel = topologyChannel.expand(-1, -1, h, w)
        topologyInput = topologyChannel[:, 0:1, :, :]                           #(batch, 1, H, W)
        return topologyInput

    def forward(self, x):        
        topologyInput = self.buildTopologyChannel(x)                            #Step1: Pre-processing: persistence on raw image        
        vggInput = torch.cat([x, topologyInput], dim=1)                         #Step2: Concatenate as input to VGG (pre-CNN)        
        return self.vgg(vggInput)                                               #Step3: Standard VGG forward



# -------------------------------------------------------------------------
# Basic CNN architecture for image classification.
# Inputs: inCh, numClasses, imgSize
# Output: class logits from a two-layer convolutional network
# -------------------------------------------------------------------------
class SimpleCnn(nn.Module):
    def __init__(self, inCh, numClasses, imgSize=(28, 28)):
        super().__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(inCh, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.imgSize = imgSize
        self.flatten_dim = self._get_flatten_dim(inCh, imgSize)

        self.fcLayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, numClasses),
        )

    def _get_flatten_dim(self, inCh, imgSize):
        with torch.no_grad():
            dummy_input = torch.zeros(1, inCh, imgSize[0], imgSize[1])
            dummy_output = self.convLayers(dummy_input)
            return dummy_output.view(1, -1).shape[1]

    def forward(self, x):
        x = self.convLayers(x)
        return self.fcLayers(x)


# -------------------------------------------------------------------------
# Build and configure a ResNet model (18/34/50) for custom input channels.
# Inputs: modelName, numClasses, inCh, weights
# Output: configured ResNet model ready for training or evaluation
# -------------------------------------------------------------------------
def buildResnet(modelName, numClasses, inCh=3, weights=None):
    if modelName == "resnet18":
        model = models.resnet18(weights=weights)
    elif modelName == "resnet34":
        model = models.resnet34(weights=weights)
    elif modelName == "resnet50":
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown ResNet variant '{modelName}'")
    
    inFeatures = model.fc.in_features
    model.fc = nn.Linear(inFeatures, numClasses)
    
    if inCh != 3:                                                           #Modify first conv for non-RGB inputs
        originalConv = model.conv1
        model.conv1 = nn.Conv2d(
            inCh, 
            originalConv.out_channels,
            kernel_size=originalConv.kernel_size,
            stride=originalConv.stride,
            padding=originalConv.padding,
            bias=originalConv.bias is not None
        )
    
    return model



# -------------------------------------------------------------------------
# Build and configure a VGG model (11/16/19) for custom input channels.
# Inputs: modelName, numClasses, inCh, weights
# Output: configured VGG model ready for training or evaluation
# -------------------------------------------------------------------------
def buildVgg(modelName, numClasses, inCh=3, weights=None):
    if modelName == "vgg11":
        model = models.vgg11(weights=weights)
    elif modelName == "vgg16":
        model = models.vgg16(weights=weights)
    elif modelName == "vgg19":
        model = models.vgg19(weights=weights)
    else:
        raise ValueError(f"Unknown VGG variant '{modelName}'")
    
    inFeatures = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(inFeatures, numClasses)
    
    if inCh != 3:                                                                   #Modify first conv for non-RGB inputs
        originalConv = model.features[0]
        model.features[0] = nn.Conv2d(
            inCh,
            originalConv.out_channels,
            kernel_size=originalConv.kernel_size,
            stride=originalConv.stride,
            padding=originalConv.padding,
            bias=originalConv.bias is not None
        )    
    return model