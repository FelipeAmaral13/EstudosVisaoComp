import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix


def get_LBP_describe(img: str, eps=1e-7):
    numPoints = 36
    radius = 12

    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(
        gray, numPoints, radius, method="uniform")

    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, numPoints + 3),
        range=(0, numPoints + 2))

    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return gray, lbp, hist


def plot_image(image: np.array, imgLBP: np.array, vecimgLBP):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title("Imagem")
    ax = fig.add_subplot(1, 3, 2)
    ax.axis('off')
    ax.imshow(imgLBP, cmap="gray")
    ax.set_title("Imagem convertida em LBP")
    ax = fig.add_subplot(1, 3, 3)
    freq, lbp, _ = ax.hist(vecimgLBP, bins=2**8)
    ax.set_ylim(0, 2)
    lbp = lbp[:-1]

    # Printar os valores do LBP quando as freqs. sao altas
    # largeTF = freq > 2
    # for x, fr in zip(lbp[largeTF], freq[largeTF]):
    #     ax.text(x, fr, "{:6.0f}".format(x), color="magenta")
    # ax.set_title("LBP histogram")
    plt.show()


def plot_cm(y_true: list, y_pred: np.array):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
    plt.show()


def plot_validation(model, X: list, Y: list):

    param_range = np.arange(1, 150, 2)
    train_scores, test_scores = validation_curve(
        model,
        X,
        Y,
        param_name='C',
        param_range=param_range,
        cv=4,
        scoring="accuracy",
        n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.subplots(1, figsize=(7, 7))
    plt.plot(param_range, train_mean, label="Treino score", color="black")
    plt.plot(param_range, test_mean, label="Teste score", color="dimgrey")

    plt.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        color="gray"
        )

    plt.fill_between(
        param_range,
        test_mean - test_std,
        test_mean + test_std,
        color="gainsboro"
        )

    plt.title("Curva Validacao SVC")
    plt.xlabel("Numeros C - Regularizacao")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()
