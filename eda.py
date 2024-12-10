from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import data_processing
import numpy as np

def main():
    target, data = data_processing.read_data("training_data_long.csv", mode = "List")
    print(data.shape)

    pca = PCA()
    pca.fit(data)
    #print(pca.explained_variance_ratio_)
    pca.components_
    transformed = pca.transform(data)
    xs=transformed[:,0]
    ys=transformed[:,1]
    zs =transformed[:,2]

    fig = plt.figure(figsize=(10, 10)) 
    Axes3D(fig) 
    ax = fig.add_subplot(projection='3d')
    fig.patch.set_facecolor('white')
    target = np.array(target)
    indices = np.where(target == 'D')
    ax.scatter(xs[indices], ys[indices], zs[indices], c="Blue", s=0.05, alpha = 0.3)
    indices = np.where(target == 'R')
    ax.scatter(xs[indices], ys[indices], zs[indices], c="Red", s=0.05, alpha = 0.3)

    plt.xlabel("First Principal Component",fontsize=14)
    plt.ylabel("Second Principal Component",fontsize=14)
    plt.legend()
    plt.show()
