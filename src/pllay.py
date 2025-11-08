import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import pdist, squareform

try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: Gudhi not available. Using simplified topological features.")


class PersistenceLandscapeLayer(nn.Module):
    def __init__(
        self,
        filtrationType="dtm",
        m0=0.05,
        kMax=3,
        resolution=50,
        useGudhi: bool = False,   # NEW: default False for speed
        maxPoints: int = 256,     # NEW: subsample point cloud
    ):
        super().__init__()
        self.filtrationType = filtrationType
        self.m0 = m0
        self.kMax = kMax
        self.resolution = resolution
        self.maxPoints = maxPoints

        # trainable weights for combining k landscapes
        self.weights = nn.Parameter(torch.ones(kMax) / kMax)

        # Decide whether to use fast fallback or real Gudhi
        self.fallbackMode = (not useGudhi) or (not GUDHI_AVAILABLE)

        if self.fallbackMode:
            if not GUDHI_AVAILABLE and useGudhi:
                print("Gudhi requested but not available; falling back to learned descriptor.")
            # Fast, learned global descriptor
            self.fallbackTransform = nn.Sequential(
                nn.Flatten(),              # (B, C, H, W) or (B, D) -> (B, D_flat)
                nn.LazyLinear(resolution), # infer in_features automatically
            )


    def computeDtmFiltration(self, points, weights=None):
        """Compute DTM filtration as described in paper (simplified)."""
        if weights is None:
            weights = torch.ones(points.shape[0], device=points.device)

        # Simplified DTM implementation
        distances = torch.cdist(points, points)        # device = points.device
        k = max(1, int(self.m0 * points.shape[0]))

        knnDistances = torch.topk(distances, k=k, largest=False, dim=1).values
        dtmValues = torch.sqrt(torch.mean(knnDistances ** 2, dim=1))
        return dtmValues

    def computePersistenceDiagram(self, points):
        """Compute persistence diagram using Vietoris–Rips complex (Gudhi, CPU)."""
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

    def persistenceDiagramToLandscapes(self, persistence, kMax=3):
        """Convert persistence diagram to persistence landscapes."""
        # Extract birth–death pairs
        birthDeathPairs = []
        for dim, (birth, death) in persistence:
            if dim in (0, 1) and death != float("inf"):
                birthDeathPairs.append((birth, death))

        # Choose a reference device (same as weights)
        device = self.weights.device

        if not birthDeathPairs:
            # empty: return zeros on correct device
            return torch.zeros(kMax, self.resolution, device=device)

        landscapes = []
        tValues = torch.linspace(0, 1, self.resolution, device=device)

        for k in range(1, kMax + 1):
            landscapeK = []
            for t in tValues:
                values = []
                for birth, death in birthDeathPairs:
                    if birth <= float(t) <= death:
                        # triangle function
                        value = max(0.0, min(float(t) - birth, death - float(t)))
                        values.append(value)

                if len(values) >= k:
                    valuesSorted = sorted(values, reverse=True)
                    landscapeK.append(valuesSorted[k - 1])
                else:
                    landscapeK.append(0.0)

            # create tensor on correct device
            landscapeK_tensor = torch.tensor(landscapeK, device=device, dtype=torch.float32)
            landscapes.append(landscapeK_tensor)

        return torch.stack(landscapes)  # (kMax, resolution)

    def forward(self, x):
        """Main forward pass implementing PLLay."""

        # Fallback: no Gudhi, use simple global descriptor (works CPU/GPU)
        if getattr(self, "fallbackMode", False):
            if x.dim() > 2:
                # (B, C, H, W) -> (B, D_flat) via Flatten inside fallbackTransform
                return self.fallbackTransform(x)
            else:
                # (B, D) already flat
                return self.fallbackTransform(x)

        batchSize = x.shape[0]
        device = x.device

        # Convert images to point clouds (stays on same device as x)
        pointClouds = self.imageToPointCloud(x)

        allLandscapeFeatures = []

        for i in range(batchSize):
            points = pointClouds[i]  # (N_points, 3) on device

            if points.shape[0] < 3:
                landscapeFeatures = torch.zeros(self.resolution, device=device)
            else:
                # 1. Compute persistence diagram (CPU via Gudhi)
                persistence = self.computePersistenceDiagram(points)

                # 2. Convert to persistence landscapes (device = self.weights.device)
                landscapes = self.persistenceDiagramToLandscapes(persistence, self.kMax)

                if landscapes.numel() > 0:
                    # 3. Weighted average landscape, all on same device
                    weightedLandscape = torch.sum(
                        self.weights.unsqueeze(1) * landscapes, dim=0
                    )
                    landscapeFeatures = weightedLandscape.to(device)
                else:
                    landscapeFeatures = torch.zeros(self.resolution, device=device)

            allLandscapeFeatures.append(landscapeFeatures)

        # (B, resolution)
        return torch.stack(allLandscapeFeatures, dim=0)

    def imageToPointCloud(self, images):
        """Convert batch of images to point clouds using pixel intensities."""
        batchPointClouds = []

        for img in images:
            if img.dim() == 3:  # (C, H, W)
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
                points = points.reshape(-1, 3)  # (H*W, 3)
            else:
                # Fallback: flatten whatever is left
                points = img.reshape(-1, img.shape[-1] if img.dim() > 1 else 1)

            # NEW: subsample to keep Gudhi manageable
            if (not getattr(self, "fallbackMode", False)) and self.maxPoints is not None:
                if points.shape[0] > self.maxPoints:
                    idx = torch.randperm(points.shape[0], device=points.device)[: self.maxPoints]
                    points = points[idx]

            batchPointClouds.append(points)

        return batchPointClouds
