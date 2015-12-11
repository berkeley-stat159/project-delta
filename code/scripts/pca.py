"""
Purpose
-------
The script performs a Principal Component Analysis on the filtered data set to
determine spatial patterns that account for the greatest amount of variability
in a time series. This requires finding the singular value decomposition of the
data matrix, which also has the advantage of providing a way to simplify the
data and filter out unwanted components.

This script should output a total of 
"""
from __future__ import absolute_import, division, print_function
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.linalg as npl
import os, sys
import pandas as pd

sys.path.append("code/utils")
from make_class import *


# Create a collection of all subject IDs and all run IDs
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
IDs = list(zip([run_ID for _ in range(16) for run_ID in run_IDs],
               [subject_ID for _ in range(3) for subject_ID in subject_IDs]))
IDs.sort()

# We perform the procedure outlined in this script for each run of each subject:
for ID in IDs:
    run, subject = ID


    # Define results directories to which to save the figures produced
    path_result = "results/run%s/pca/sub%s/" % ID
    try:
        os.makedirs(path_result)
    except OSError:
        if not os.path.isdir(path_result):
            raise


    # Extract the data of interest
    data = ds005(subject, run).filtered.data

    # Define some useful variables for later on
    volume_shape, num_volumes = data.shape[:3], data.shape[3]
    
    # Reshape array for easy manipulation of voxels
    data2d = np.reshape(data, (-1, num_volumes))

    # Subtract the mean of first dimension 
    data_red = data2d - np.mean(data2d, axis=0)

    # Compute covariance matrix
    cov_matrix = data_red.T.dot(data_red)

    # Find the singular value decomposition of the covariance matrix
    U1, S, U2 = npl.svd(cov_matrix)


    # We will now produce a graphical display of each dataset component's
    # variance to determine how many components should be retained.
    # For reference, a Scree plot shows the fraction of the total variance that
    # is explained or represented by each principal component (the eigenvalue!)
    eigenvalues = S ** 2 / np.cumsum(S)[-1]
    fig = plt.figure(figsize = (8, 5))
    singular_values = np.arange(num_volumes) + 1
    plt.plot(singular_values, eigenvalues, "ro-", linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    leg = plt.legend(["Eigenvalues from SVD"], loc="best", borderpad=0.3, 
                     shadow=False, markerscale=0.4,
                     prop=matplotlib.font_manager.FontProperties(size='small'))
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    #plt.show()
    plt.savefig(path_result + "scree_plot_1.png")
    plt.close()


    # We repeat the process a second time, the idea being that some linear
    # combination of the loadings from second dimension will have a square sum
    # of 1. This particular linear combination yields the highest variance, and
    # we seek to find it.
    # Subtract the mean of the second dimension
    data_red_x2 = data_red - np.mean(data_red, axis=1).reshape((-1, 1))
    
    # Compute the covariance matrix
    cov_matrix_red_x2 = data_red_x2.T.dot(data_red_x2)

    # Find the singular value decomposition of the covariance matrix
    U1, S, U2 = npl.svd(cov_matrix_red_x2)


    # Again, let's produce a figure. This time, you can clearly see the variance
    # explained by the first component and then the additional variance for each
    # subsequent component. Projection of the observations onto this vector
    # yields the highest possible variance among observations.
    eigenvalues = S ** 2 / np.cumsum(S)[-1]
    fig = plt.figure(figsize=(8, 5))
    singular_values = np.arange(num_volumes) + 1
    plt.plot(singular_values, eigenvalues, 'ro-', linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    leg = plt.legend(["Eigenvalues from SVD"], loc="best", borderpad=0.3, 
                     shadow=False, markerscale=0.4,
                     prop=matplotlib.font_manager.FontProperties(size="small"))
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    #plt.show()
    plt.savefig(path_result + "scree_plot_2.png")
    plt.close()


    # Save the eigenvalues to a plaintext file
    pca = PCA(n_components=5)
    pca.fit(cov_matrix_red_x2)
    np.savetxt(path_result + "eigenvalues.txt", pca.explained_variance_ratio_)


    # Fit the original model and apply to it the dimensionality reduction
    model = pca.fit_transform(cov_matrix_red_x2)
    np.save(path_result + "model.txt", model)