import histomicstk as htk
import matplotlib.pyplot as plt
import numpy as np
import pygco as gc
import scipy as sp
import scipy.ndimage.measurements as ms
from scipy.stats import multivariate_normal as mvn
import skimage.io as io


def multiway_cut():

    # return label image and seeds

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

    # constrained log filtering=== - generate R_{N}(x,y)
    Nuclei = UN1.Stains[::2, ::2, 0].astype(dtype=np.uint8)
    Tau, Foreground, Background = htk.PoissonMixture(Nuclei)
    Mask = binary_cut(Foreground, Background, Nuclei, Sigma=50)

    Response = htk.cLoG(Nuclei, Mask, SigmaMin=4*1.414, SigmaMax=7*1.414)
    Label, Seeds, Max = htk.MaxClustering(Response.copy(), Mask, 10)
    Filtered = htk.FilterLabel(Label, 4, 80, None)

    # multiway graph cut refinement of max-clustering segmentation
    _multiway_refine(Nuclei, Response, Filtered, Background=5e-1,
                     Smoothness=1e-5)

    return


def binary_cut(Foreground, Background, I=None, Sigma=None):
    """Performs a binary graph-cut optimization for region masking, given
    foreground and background probabilities for image and

    Parameters
    ----------
    I : array_like
        An intensity image used for calculating the continuity term. If not
        provided, only the data term from the foreground/background model will
        drive the graph-cut optimization. Default value = None.
    Foreground : array_like
        Intensity image the same size as 'I' with values in the range [0, 1]
        representing probabilities that each pixel is in the foreground class.
    Background : array_like
        Intensity image the same size as 'I' with values in the range [0, 1]
        representing probabilities that each pixel is in the background class.
    Sigma : double
        Parameter controling exponential decay rate of continuity term. Units
        are intensity values. Recommended ~10% of dynamic range of image.
        Defaults to 25 if 'I' is type uint8 (range [0, 255]), and 0.1 if type
        is float (expected range [0, 1]). Default value = None.

    Notes
    -----
    The graph cutting algorithm is sensitive to the magnitudes of weights in
    the data variable 'D'. Small probabilities will be magnified by the
    logarithm in formulating this cost, and so we add a small positive value
    to each in order to scale them appropriately.

    Returns
    -------
    Mask : array_like
        A boolean-type binary mask indicating the foreground regions.

    See Also
    --------
    PoissonMixture

    References
    ----------
    .. [1] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
    of Cell Nuclei in Histopathology Images" in IEEE Transactions on Biomedical
    Engineering,vol.57,no.4,pp.847-52, 2010.
    """

    # generate a small number for conditioning calculations
    Small = np.finfo(np.float).eps

    # formulate unary data costs
    D = np.stack((-np.log(Background + Small) / -np.log(Small),
                 -np.log(Foreground + Small) / -np.log(Small)),
                 axis=2)

    # formulate pairwise label costs
    Pairwise = 1 - np.eye(2)

    # calculate edge weights between pixels if argument 'I' is provided
    if I is not None:
        if Sigma is None:
            if issubclass(I.dtype.type, np.float):
                Sigma = 0.1
            elif I.dtype.type == np.uint8:
                Sigma = 25

        # formulate vertical and horizontal costs
        H = np.exp(-np.abs(I[:, 0:-1].astype(np.float) -
                   I[:, 1:].astype(np.float)) / (2 * Sigma**2))
        V = np.exp(-np.abs(I[0:-1, :].astype(np.float) -
                   I[1:, :].astype(np.float)) / (2 * Sigma**2))

        # cut the graph with edge information from image
        Mask = gc.cut_grid_graph(D, Pairwise, V, H, n_iter=-1,
                                 algorithm='expansion')

    else:
        # cut the graph without edge information from image
        Mask = gc.cut_grid_graph_simple(D, Pairwise, n_iter=-1,
                                        algorithm='expansion')

    # reshape solution to image size and return
    return Mask.reshape(Foreground.shape[0], Foreground.shape[1]) == 1



def constrained_log():

    # return response

    return


def max_clustering():

    # return label image

    return


def _multiway_refine(I, Response, Label, Background=5e-1, Smoothness=1e-5):

    # initialize output image
    Refined = np.zeros(Label.shape, dtype=np.uint32)

    # initialize cell count
    Total = 0

    # identify connected components
    Components, N = ms.label(Label > 0)

    # get locations of connected components
    Locations = ms.find_objects(Components)

    # i = 7, 16, 37 - examples of clumped objects

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
            for k in np.arange(len(Indices)):

                # define x, y coordinates of nucleus
                cY, cX = np.nonzero(Component == k+1)

                # model each nucleus with gaussian and add to component 'j'
                Mean, Cov = _gaussian_model(Response[i-1], cX, cY)

                # define multivariate normal for object k
                Model = mvn(Mean.flatten(), Cov.squeeze())

                # add multivariate normal to channel 'j' of D
                D[:, :, k+1] = np.maximum(D[:, :, k+1], Model.pdf(Pos))

        # add background probability
        D[:, :, 0] = Background

        # score probabilities
        for j in np.arange(1, D.shape[3]):
            D[:, :, j] = 1 - D[:, :, j]

        # formulate image-based edge costs
        Horizontal = np.exp(-np.abs(I[:, 0:-1] - I[:, 1:]))
        Vertical = np.exp(-np.abs(I[0:-1, :] - I[1:, :]))

        # formulate label cost
        X, Y = np.mgrid[:D.shape[3], :D.shape[3]]
        V = Smoothness * np.float_(np.abs(X-Y))

        # cut the graph
        Cut = gc.cut_grid_graph(D, V, Vertical, Horizontal,
                                n_iter=-1, algorithm='swap')

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
    Coords = np.hstack((Y, X))

    # estimate weighted mean
    Mean = np.dot(Weights.transpose(), Coords) / np.sum(Weights)

    # estimate weighted covariance
    MeanVec = np.dot(np.ones(Weights.shape), Mean)
    Covariance = np.dot((Coords - MeanVec).transpose(),
                        np.hstack((Weights, Weights)) * (Coords-MeanVec))
    Covariance = Covariance / (np.sum(Weights)-1)

    return Mean, Covariance
