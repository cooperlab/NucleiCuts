import histomicstk as htk
import matplotlib.pyplot as plt
import numpy as np
import pygco as gc
import scipy.ndimage.measurements as ms
from scipy.stats import multivariate_normal as mvn
import skimage.io as io


def multiway_cut():

    W = np.array([[0.650, 0.072, 0],
                  [0.704, 0.990, 0],
                  [0.286, 0.105, 0]])
    Directory = '/Users/lcoop22/Desktop/Detection/'

    Standard = io.imread(Directory + 'Hard.png')[:, :, :3]
    I1 = io.imread(Directory + 'Easy1.png')[:, :, :3]

    # calculate mean, SD of standard image in LAB color space
    LABStandard = htk.RudermanLABFwd(Standard)
    m = Standard.shape[0]
    n = Standard.shape[1]
    Mu = LABStandard.sum(axis=0).sum(axis=0) / (m*n)
    LABStandard[:, :, 0] = LABStandard[:, :, 0] - Mu[0]
    LABStandard[:, :, 1] = LABStandard[:, :, 1] - Mu[1]
    LABStandard[:, :, 2] = LABStandard[:, :, 2] - Mu[2]
    Sigma = ((LABStandard*LABStandard).sum(axis=0).sum(axis=0) / (m*n-1))**0.5
    LABStandard[:, :, 0] = LABStandard[:, :, 0] / Sigma[0]
    LABStandard[:, :, 1] = LABStandard[:, :, 1] / Sigma[1]
    LABStandard[:, :, 2] = LABStandard[:, :, 2] / Sigma[2]

    # normalize input images
    N1 = htk.ReinhardNorm(I1, Mu, Sigma)

    # color deconvolutions - normalized
    UN1 = htk.ColorDeconvolution(N1, W)

    # generate poisson mixture model for hematoxylin channel
    Nuclei = UN1.Stains[::2, ::2, 0].astype(dtype=np.uint8)
    Tau, Foreground, Background = htk.PoissonMixture(Nuclei)

    # perform a graph cut to distinguish foreground and background
    Mask = binary_cut(Foreground, Background, Nuclei, Sigma=50)

    # constrained log filtering
    Response = htk.cLoG(Nuclei, Mask, SigmaMin=4*1.414, SigmaMax=7*1.414)

    # cluster pixels to constrained log maxima
    Label, Seeds, Max = htk.MaxClustering(Response.copy(), Mask, 10)

    # cleanup label image - split, then open by area and width
    Label = htk.SplitLabel(Label)
    Label = htk.AreaOpenLabel(Label, 20)
    Label = htk.WidthOpenLabel(Label, 5)

    # multiway graph cut refinement of max-clustering segmentation
    Label = _multiway_refine(Nuclei, Response, Label, Background=5e-1,
                             Smoothness=1e-5)

    return Label


def _multiway_refine(I, Response, Label, Background=5e-1, Smoothness=1e-5):

    # initialize output image
    Refined = np.zeros(Label.shape, dtype=np.uint32)

    # initialize cell count
    Total = 0

    # identify connected components
    Components, N = ms.label(Label > 0)

    # get locations of connected components
    Locations = ms.find_objects(Components)

    # i = 5, 14, 35

    # process each connected component containing possibly multiple nuclei
    for i in np.arange(1, N+1):

        # generate mask of connected component
        Mask = Components == i

        # extract label image and mask
        Component = Label[Locations[i-1]].copy()
        ComponentMask = Mask[Locations[i-1]]

        # zero out labels not in component
        Component[~ComponentMask] = 0

        # condense label image
        Component = htk.CondenseLabel(Component)

        # generate region adjacency graph
        Adjacency = htk.LabelRegionAdjacency(Component, Neighbors=4)

        # generate region adjacencey graph
        RAG = htk.GraphColorSequential(Adjacency)

        # layer region adjancency graph
        RAG = htk.RegionAdjacencyLayer(RAG)

        # generate bounding box patch for graph cutting problem
        D = np.zeros((Component.shape[0], Component.shape[1], len(RAG)+1),
                     dtype=np.float)
        X, Y = np.meshgrid(range(0, Component.shape[1]),
                           range(0, Component.shape[0]))
        Pos = np.empty(X.shape + (2,))
        Pos[:, :, 1] = X
        Pos[:, :, 0] = Y

        # for each color in rag
        for j in np.arange(1, np.max(RAG)+1):

            # get indices of cells to model in color 'j'
            Indices = np.nonzero(RAG == j)[0]

            # for each nucleus in color
            for k in Indices:

                # define x, y coordinates of nucleus
                cY, cX = np.nonzero(Component == k+1)

                # model each nucleus with gaussian and add to component 'j'
                Mean, Cov = _gaussian_model(Response[Locations[i-1]], cX, cY)

                # define multivariate normal for object k
                Model = mvn(Mean.flatten(), Cov.squeeze())

                # add multivariate normal to channel 'j' of D
                D[:, :, j] = np.maximum(D[:, :, j], Model.pdf(Pos))

        # add background probability
        D[:, :, 0] = Background

        # score probabilities
        for j in np.arange(1, D.shape[2]):
            D[:, :, j] = 1 - D[:, :, j]

        # formulate image-based edge costs
        Patch = I[Locations[i-1]].astype(np.float)
        Horizontal = np.exp(-np.abs(Patch[:, 0:-1] - Patch[:, 1:]))
        Vertical = np.exp(-np.abs(Patch[0:-1, :] - Patch[1:, :]))

        # formulate label cost
        X, Y = np.mgrid[:D.shape[2], :D.shape[2]]
        V = Smoothness * np.float_(np.abs(X-Y))

        # cut the graph
        Cut = gc.cut_grid_graph(D, V, Vertical, Horizontal,
                                n_iter=-1, algorithm='swap')

        # reshape the cut to the original component patch size
        Cut = Cut.reshape(Component.shape[0], Component.shape[1])

        # update the values in the cut
        Cut = Cut + Total

        # embed the resulting cut into the output label image
        Refined[Mask] = Cut[ComponentMask]

        # update cell count
        Total = Total + Cut.max()

    return Refined


def _gaussian_model(I, X, Y):
    """Generates a two-dimensional gaussian model of a segmented nucleus using
    weighted estimation.

    Parameters
    ----------
    I : array_like
        Image values used in weighted estimation of gaussian. Constrained
        laplacian of gaussian values are used in this case. Do not need to be
        scaled / normalized.
    X : array_like
        Horizontal coordinates of the object in 'I'.
    Y : array_like
        Vertical coordinates of the object in 'I'.

    Returns
    -------
    Mean : array_like
        2 element vector of object mean.
    Covariance : array_like
        2 x 2 matrix of object covariances.
    """

    # get image values at object locations for pixel weighting
    Weights = I[Y, X]

    # stack object coordinates into matrix
    Coords = np.vstack((Y, X))

    # estimate weighted mean
    Mean = np.dot(Coords, Weights) / np.sum(Weights)

    # estimate weighted covariance
    MeanVec = np.tile(Mean, (Coords.shape[1], 1)).transpose()
    Covariance = np.dot((Coords - MeanVec),
                        (np.vstack((Weights, Weights)) *
                        (Coords-MeanVec)).transpose())
    Covariance = Covariance / (np.sum(Weights)-1)

    return Mean, Covariance
