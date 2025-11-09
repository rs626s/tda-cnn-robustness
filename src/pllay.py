from scipy.spatial.distance import pdist, squareform
import torch.nn as nn
import numpy as np
import torch

try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: Gudhi not available. Using simplified topological features.")


# -------------------------------------------------------------------------
# PersistenceLandscapeLayer: Implements a persistence landscape layer (PLLay)
# for topological feature extraction using either Gudhi or a learned fallback.
# Inputs: (B, C, H, W) images
# Output: (B, resolution) persistence landscape features
# -------------------------------------------------------------------------
class PersistenceLandscapeLayer(nn.Module):
    def __init__(
        self,
        filtrationType="dtm",
        m0=0.05,
        kMax=3,
        resolution=50,
        useGudhi: bool = False,                                                         #default False for speed
        maxPoints: int = 256,                                                           #subsample point cloud
    ):
        super().__init__()
        self.filtrationType = filtrationType
        self.m0 = m0
        self.kMax = kMax
        self.resolution = resolution
        self.maxPoints = maxPoints        
        self.weights = nn.Parameter(torch.ones(kMax) / kMax)                            #trainable weights for combining k landscapes        
        self.fallbackMode = (not useGudhi) or (not GUDHI_AVAILABLE)                     #Decide whether to use fast fallback or real Gudhi

        if self.fallbackMode:
            if not GUDHI_AVAILABLE and useGudhi:
                print("Gudhi requested but not available; falling back to learned descriptor.")            
            self.fallbackTransform = nn.Sequential(                                     #Fast, learned global descriptor
                nn.Flatten(),                                                           #(B, C, H, W) or (B, D) -> (B, D_flat)
                nn.LazyLinear(resolution),                                              #infer in_features automatically
            )

    
    # ---------------------------------------------------------------------
    # Compute DTM (Distance-to-Measure) filtration for a point cloud.
    # Inputs: points, optional weights
    # Output: DTM scalar values per point
    # ---------------------------------------------------------------------
    def computeDtmFiltration(self, points, weights=None):
        if weights is None:
            weights = torch.ones(points.shape[0], device=points.device)
        distances = torch.cdist(points, points)                                         #device = points.device
        k = max(1, int(self.m0 * points.shape[0]))
        knnDistances = torch.topk(distances, k=k, largest=False, dim=1).values
        dtmValues = torch.sqrt(torch.mean(knnDistances ** 2, dim=1))
        return dtmValues
    
    
    # ---------------------------------------------------------------------
    # Compute persistence diagram using Vietoris–Rips complex via Gudhi.
    # Inputs: point cloud (N_points, D)
    # Output: list of (birth, death) pairs
    # ---------------------------------------------------------------------
    def computePersistenceDiagram(self, points):
        if not GUDHI_AVAILABLE or points.shape[0] < 3:
            return []

        pointsNp = points.detach().cpu().numpy()
        try:
            ripsComplex = gd.RipsComplex(points=pointsNp, max_edge_length=2.0)
            simplexTree = ripsComplex.create_simplex_tree(max_dimension=2)
            persistence = simplexTree.persistence()
            return persistence
        except Exception as e:
            print(f"Persistence computation failed: {e}")
            return []
    
    
    # ---------------------------------------------------------------------
    # Convert persistence diagram into persistence landscapes.
    # Inputs: persistence diagram, kMax (number of layers)
    # Output: tensor (kMax, resolution)
    # ---------------------------------------------------------------------
    def persistenceDiagramToLandscapes(self, persistence, kMax=3):
        birthDeathPairs = []
        for dim, (birth, death) in persistence:
            if dim in (0, 1) and death != float("inf"):
                birthDeathPairs.append((birth, death))
        
        device = self.weights.device                                            #Choose a reference device (same as weights)

        if not birthDeathPairs:
            return torch.zeros(kMax, self.resolution, device=device)

        landscapes = []
        tValues = torch.linspace(0, 1, self.resolution, device=device)

        for k in range(1, kMax + 1):
            landscapeK = []
            for t in tValues:
                values = []
                for birth, death in birthDeathPairs:
                    if birth <= float(t) <= death:
                        value = max(0.0, min(float(t) - birth, death - float(t)))
                        values.append(value)

                if len(values) >= k:
                    valuesSorted = sorted(values, reverse=True)
                    landscapeK.append(valuesSorted[k - 1])
                else:
                    landscapeK.append(0.0)

            landscapeK_tensor = torch.tensor(landscapeK, device=device, dtype=torch.float32)
            landscapes.append(landscapeK_tensor)

        return torch.stack(landscapes)                                                              #(kMax, resolution)
    
    
    # ---------------------------------------------------------------------
    # Main forward pass for PLLay.
    # Inputs: input tensor (B, C, H, W) or (B, D)
    # Output: topological landscape feature tensor (B, resolution)
    # ---------------------------------------------------------------------
    def forward(self, x):
        if getattr(self, "fallbackMode", False):
            if x.dim() > 2:                
                return self.fallbackTransform(x)                                    #(B, C, H, W) -> (B, D_flat) via Flatten inside fallbackTransform
            else:                
                return self.fallbackTransform(x)                                    #(B, D) already flat

        batchSize = x.shape[0]
        device = x.device        
        pointClouds = self.imageToPointCloud(x)                                     #Convert images to point clouds (stays on same device as x)
        allLandscapeFeatures = []

        for i in range(batchSize):
            points = pointClouds[i]                                                 #(N_points, 3) on device
            if points.shape[0] < 3:
                landscapeFeatures = torch.zeros(self.resolution, device=device)
            else:                
                persistence = self.computePersistenceDiagram(points)                        #Step1: Compute persistence diagram (CPU via Gudhi)                
                landscapes = self.persistenceDiagramToLandscapes(persistence, self.kMax)    #Step2: Convert to persistence landscapes (device = self.weights.device)
                if landscapes.numel() > 0:                    
                    weightedLandscape = torch.sum(                                          #Step3: Weighted average landscape, all on same device
                        self.weights.unsqueeze(1) * landscapes, dim=0
                    )
                    landscapeFeatures = weightedLandscape.to(device)
                else:
                    landscapeFeatures = torch.zeros(self.resolution, device=device)

            allLandscapeFeatures.append(landscapeFeatures)

        return torch.stack(allLandscapeFeatures, dim=0)
    
    
    # ---------------------------------------------------------------------
    # Convert image batch into 3D point clouds (x, y, intensity).
    # Inputs: images (B, C, H, W)
    # Output: list of point clouds [(H*W, 3), ...]
    # ---------------------------------------------------------------------
    def imageToPointCloud(self, images):
        batchPointClouds = []
        for img in images:
            if img.dim() == 3:                                                              #(C, H, W)
                c, h, w = img.shape
                yCoords, xCoords = torch.meshgrid(
                    torch.arange(h, device=img.device),
                    torch.arange(w, device=img.device),
                    indexing="ij",
                )
                if c == 1:
                    intensities = img[0]
                else:
                    intensities = img.mean(dim=0)

                points = torch.stack(
                    [
                        xCoords.float() / w,
                        yCoords.float() / h,
                        intensities.float(),
                    ],
                    dim=-1,
                )
                points = points.reshape(-1, 3)                                              #(H*W, 3)
            else:                
                points = img.reshape(-1, img.shape[-1] if img.dim() > 1 else 1)             #Fallback: flatten whatever is left

            if (not getattr(self, "fallbackMode", False)) and self.maxPoints is not None:
                if points.shape[0] > self.maxPoints:
                    idx = torch.randperm(points.shape[0], device=points.device)[: self.maxPoints]
                    points = points[idx]
            batchPointClouds.append(points)

        return batchPointClouds
