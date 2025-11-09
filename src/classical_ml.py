from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog
from sklearn.svm import SVC
from tqdm import tqdm
import torchvision.datasets as dsets
import torchvision.transforms as T
import matplotlib.pyplot as plt
import argparse, os, json, time
import numpy as np

# -------------------------------------------------------------------------
# Load and return train/test splits for the selected dataset.
# Inputs: name ('mnist', 'fashion', 'cifar10'), max_samples (optional)
# Output: Xtr, ytr, Xte, yte (NumPy arrays of images and labels)
# -------------------------------------------------------------------------
def load_dataset(name, max_samples=None):
    if name=="mnist":
        ds_train = dsets.MNIST("./data", train=True, download=True, transform=T.ToTensor())
        ds_test = dsets.MNIST("./data", train=False, download=True, transform=T.ToTensor())
    elif name=="fashion":
        ds_train = dsets.FashionMNIST("./data", train=True, download=True, transform=T.ToTensor())
        ds_test = dsets.FashionMNIST("./data", train=False, download=True, transform=T.ToTensor())
    elif name=="cifar10":
        ds_train = dsets.CIFAR10("./data", train=True, download=True, transform=T.ToTensor())
        ds_test = dsets.CIFAR10("./data", train=False, download=True, transform=T.ToTensor())
    else:
        raise ValueError("dataset not supported")
    Xtr = ds_train.data.numpy()
    ytr = np.array(ds_train.targets)
    Xte = ds_test.data.numpy()
    yte = np.array(ds_test.targets)
    if max_samples:
        Xtr, ytr = Xtr[:max_samples], ytr[:max_samples]
    return Xtr, ytr, Xte, yte


# -------------------------------------------------------------------------
# Compute Histogram of Oriented Gradients (HOG) features for images.
# Inputs: images (array of shape [N, H, W] or [N, H, W, C])
# Output: NumPy array of HOG feature vectors
# -------------------------------------------------------------------------
def feat_hog(images):
    feats = []
    for img in tqdm(images, desc="HOG"):
        if img.ndim==3 and img.shape[-1]==3:            
            img = img.mean(axis=-1)                                                                         # convert to grayscale simple average
        f = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        feats.append(f)
    return np.array(feats)


# -------------------------------------------------------------------------
# Extract PCA-based features by flattening and reducing image dimensions.
# Inputs: images (array [N, H, W, C] or [N, H, W]), n_components (default: 100)
# Output: NumPy array of PCA-transformed features
# -------------------------------------------------------------------------
def feat_pca(images, n_components=100):
    N = images.shape[0]
    flat = images.reshape(N, -1).astype(np.float32)/255.0
    scaler = StandardScaler(with_mean=True, with_std=True)
    flat = scaler.fit_transform(flat)
    p = PCA(n_components=n_components, random_state=42)
    return p.fit_transform(flat)


# -------------------------------------------------------------------------
# Execute a classical ML experiment with HOG/PCA features and SVM/RF classifier.
# Inputs: dataset ('mnist', 'fashion', 'cifar10'), feature ('hog' or 'pca'),
#         clf ('svm' or 'rf'), max_samples, out_dir (default: 'outputs')
# Output: metrics JSON and confusion matrix plot (.png)
# -------------------------------------------------------------------------
def run(dataset, feature, clf, max_samples, out_dir="outputs"):
    Xtr, ytr, Xte, yte = load_dataset(dataset, max_samples=max_samples)
    if feature=="hog":
        Ftr = feat_hog(Xtr); Fte = feat_hog(Xte)
    elif feature=="pca":
        Ftr = feat_pca(Xtr); Fte = feat_pca(Xte)
    else:
        raise ValueError("feature must be hog or pca")
    if clf=="svm":
        model = SVC(kernel="rbf", C=10, gamma="scale")
    elif clf=="rf":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError("clf must be svm or rf")
    t0 = time.time()
    model.fit(Ftr, ytr)
    train_time = time.time()-t0
    ypred = model.predict(Fte)
    acc = accuracy_score(yte, ypred)

    # Save confusion matrix
    cm = confusion_matrix(yte, ypred)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"CM {dataset} {feature}+{clf}")
    plt.colorbar()
    plt.tight_layout()
    run_id = f"classical_{dataset}_{feature}_{clf}"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{run_id}_cm.png"), bbox_inches='tight')
    plt.close(fig)

    # Save metrics
    metrics = {"dataset": dataset, "feature": feature, "clf": clf,
               "test_acc": float(acc), "train_time_sec": float(train_time),
               "samples_train": int(Xtr.shape[0])}
    with open(os.path.join(out_dir, f"{run_id}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

# -------------------------------------------------------------------------
# Entry point: parse command-line arguments and run the selected experiment.
# Inputs (CLI): --dataset, --feature, --clf, --max-samples
# Action: calls run() with provided parameters
# -------------------------------------------------------------------------
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mnist","fashion","cifar10"])
    ap.add_argument("--feature", required=True, choices=["hog","pca"])
    ap.add_argument("--clf", required=True, choices=["svm","rf"])
    ap.add_argument("--max-samples", type=int, default=10000)
    args = ap.parse_args()
    run(args.dataset, args.feature, args.clf, args.max_samples)
