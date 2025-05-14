import struct
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os

# reference taken from https://www.kaggle.com/datasets/hojjatk/mnist-dataset/code
class MnistDataloader:
    def __init__(self):
        pass

    def read_images_labels(self, img_fp, lbl_fp):
        with open(lbl_fp, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            assert magic == 2049, "Invalid label file"
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        with open(img_fp, 'rb') as f:
            magic, num2, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, "Invalid image file"
            data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num2, rows * cols)
        return images, labels

    def load_data(self, train_img, train_lbl, test_img, test_lbl):
        xtr, ytr = self.read_images_labels(train_img, train_lbl)
        xte, yte = self.read_images_labels(test_img, test_lbl)
        return (xtr, ytr), (xte, yte)
    
def PCA(X_train, X_val, X_test, n_components=5):
    mean = X_train.mean(axis=0)
    Xc = X_train - mean
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    top = vecs[:, np.argsort(vals)[::-1][:n_components]]
    return Xc.dot(top), (X_val - mean).dot(top), (X_test - mean).dot(top)

def predict_stump(X, feature, threshold, left_label, right_label):
    return np.where(X[:, feature] <= threshold, left_label, right_label)

#  Function to find best stump
def get_best_stump(X, y, w, cuts):
    n, d = X.shape
    error = np.inf
    ans = None
    for j in range(d):
        for thr in cuts[j]:

            pred = np.where(X[:, j] <= thr, 1, -1)
            err = np.sum(w * (pred != y))

            err_inv = np.sum(w * (pred == y))
            if err_inv < err:
                err, lp, rp = err_inv, -1, 1
            else:
                lp, rp = 1, -1
            if err < error:
                error = err
                ans = (j, thr, lp, rp)
                
    feature, threshold, left_label, right_label = ans
    return feature, threshold, left_label, right_label, error

def compute_beta(err):
    eps = 1e-10  # small number to avoid log(0)
    err = np.clip(err, eps, 1 - eps)  
    return 0.5 * np.log((1 - err) / err)


#  Function to update weights 
def update_weights(w, beta, y, pred):
    w = w * np.exp(-beta * y * pred)
    return w / w.sum()


def adaboost(X_train, y_train, X_val, y_val, X_test, y_test, rounds, num_cuts=3):
    n, d = X_train.shape
    w = np.ones(n) / n  # initialize weights

    # prepare cuts for each feature
    cuts = [
        np.linspace(X_train[:, j].min(), X_train[:, j].max(), num_cuts + 2)[1:-1]
        for j in range(d)
    ]


    F_tr = np.zeros(n)
    F_val = np.zeros(len(y_val)) if X_val is not None else None
    F_te = np.zeros(len(y_test))

    train_loss, val_loss, test_loss, train_error = [], [], [], []

    for t in range(rounds):
        # fit best stump
        f, thr, lp, rp, err = get_best_stump(X_train, y_train, w, cuts)
        beta = compute_beta(err)
        pred_tr = predict_stump(X_train, f, thr, lp, rp)    # predict training set
        w = update_weights(w, beta, y_train, pred_tr)       # update weights
        print (w)


        F_tr += beta * pred_tr
        F_val += beta * predict_stump(X_val, f, thr, lp, rp)
        F_te += beta * predict_stump(X_test, f, thr, lp, rp)

        # record metrics
        tr_err = np.mean(np.sign(F_tr) != y_train)
        te_err = np.mean(np.sign(F_te) != y_test)
        train_error.append(tr_err)      # training error
        train_loss.append(tr_err)       # training loss
        test_loss.append(te_err)        # test loss
                    
        va_err = np.mean(np.sign(F_val) != y_val)
        val_loss.append(va_err)

    return train_loss, val_loss, test_loss, train_error

# Load MNIST dataset
input_path = "./mnist-dataset"
train_img = os.path.join(input_path,"train-images-idx3-ubyte","train-images-idx3-ubyte")
train_lbl = os.path.join(input_path,"train-labels-idx1-ubyte","train-labels-idx1-ubyte")
test_img  = os.path.join(input_path,"t10k-images-idx3-ubyte","t10k-images-idx3-ubyte")
test_lbl  = os.path.join(input_path,"t10k-labels-idx1-ubyte","t10k-labels-idx1-ubyte")


train_img = r"train-images.idx3-ubyte"
train_lbl = r"train-labels.idx1-ubyte"
test_img = r"t10k-images.idx3-ubyte"
test_lbl = r"t10k-labels.idx1-ubyte"


(X_full, y_full), (X_test_full, y_test_full) = MnistDataloader().load_data(train_img, train_lbl, test_img, test_lbl)

# keep only digits 0,1
mask_train = np.isin(y_full, [0,1])
mask_test = np.isin(y_test_full, [0,1])
X_tr, y_tr = X_full[mask_train], y_full[mask_train]
X_te, y_te = X_test_full[mask_test], y_test_full[mask_test]


np.random.seed(79)  

# Get indices for each class
ind = []
for i in [0, 1]:
    i_ind = np.where(y_tr == i)[0]
    np.random.shuffle(i_ind)
    ind.append(i_ind[:1000])

# Combine and shuffle indices
idx = np.hstack(ind)
np.random.shuffle(idx)

# Split into training and val
cut = int(len(idx) * 0.8)   # 80% for training, 20% for validation
X_train = X_tr[idx[:cut]]
Y_train = y_tr[idx[:cut]]
X_val = X_tr[idx[cut:]]
y_val = y_tr[idx[cut:]]


# PCA → 5 dimensions 
X_train_PCA, X_val_PCA, X_test_PCA = PCA(X_train, X_val, X_te, n_components=5)

#  4) Encode labels as ±1 
y_train_PCA  = np.where(Y_train  == 1, 1, -1)
y_val_PCA = np.where(y_val == 1, 1, -1)
y_test_PCA  = np.where(y_te == 1, 1, -1)

train_loss, val_loss, test_loss, train_error = adaboost(X_train_PCA, y_train_PCA, X_val_PCA, y_val_PCA, X_test_PCA, y_test_PCA, rounds=100, num_cuts=3)

#  6) Plots 
rounds = np.arange(1, len(train_loss) + 1)

plt.figure(figsize=(8,5))
plt.plot(rounds, train_loss, label="Train Loss")
plt.plot(rounds, val_loss,   label="Val Loss")
plt.plot(rounds, test_loss,  label="Test Loss")
plt.xlabel("Boosting Round")
plt.ylabel("0–1 Loss")
plt.title("Train/Val/Test Loss vs Rounds")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(rounds, train_error, label="Train Error")
plt.xlabel("Boosting Round")
plt.ylabel("0–1 Error")
plt.title("Training Error vs Rounds")
plt.show()

#  7) Final Test Accuracy 
print(f"Final Test Accuracy: {1 - test_loss[-1]:.4f}")