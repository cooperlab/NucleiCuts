from binary_cut import binary_cut
import histomicstk.segmentation.label as lb
import histomicstk.segmentation.nuclear as nc
import histomicstk.segmentation as sg
import histomicstk.utils as ut
import histomicstk.filters.shape as sh
import numpy as np
import pygco as gc
import scipy.ndimage.measurements as ms
from scipy.stats import multivariate_normal as mvn


def nuclear_cut(I, Sigma=25, SigmaMin=4*(2**0.5), SigmaMax=7*(2**0.5), r=10,
                MinArea=20, MinWidth=5, Background=1e-4, Smoothness=1e-4):

    # generate poisson mixture model for nuclei
    Tau, Fg, Bg = ut.PoissonMixture(I)

    # perform a graph cut to distinguish foreground and background
    Mask = binary_cut(Fg, Bg, I, Sigma)

    # constrained log filtering
    Response = sh.cLoG(I, Mask, SigmaMin, SigmaMax)

    # cluster pixels to constrained log maxima
    Label, Seeds, Max = nc.MaxClustering(Response.copy(), Mask, r)

    # cleanup label image - split, then open by area and width
    Label = lb.SplitLabel(Label)
    Label = lb.AreaOpenLabel(Label, MinArea)
    Label = lb.WidthOpenLabel(Label, MinWidth)

    # multiway graph cut refinement of max-clustering segmentation
    Refined = _multiway_refine(I, Response, Label, Background, Smoothness)

    return Label, Refined


def _multiway_refine(I, Response, Label, Background=1e-4, Smoothness=1e-4):

    # initialize output image
    Refined = Label.copy()

    # initialize cell count
    Total = 0

    # identify connected components
    Components, N = ms.label(Label > 0)

    # get locations of connected components
    Locations = ms.find_objects(Components)

    # process each connected component containing possibly multiple nuclei
    for i in np.arange(1, N+1):

        # extract label image and component mask
        Component = Label[Locations[i-1]].copy()
        ComponentMask = Components[Locations[i-1]] == i

        # zero out labels not in component
        Component[~ComponentMask] = 0

        # condense label image
        Component = lb.CondenseLabel(Component)

        # determine if more than one label value exists in Component
        if(Component.max() > 1):

            # generate region adjacency graph
            Adjacency = sg.LabelRegionAdjacency(Component, Neighbors=4)

            # layer region adjancency graph
            Adjacency = sg.RegionAdjacencyLayer(Adjacency)

            # generate region adjacencey graph
            RAG = sg.GraphColorSequential(Adjacency)

            # generate bounding box patch for graph cutting problem
            D = np.zeros((Component.shape[0], Component.shape[1],
                          np.max(RAG)+1), dtype=np.float)
            X, Y = np.meshgrid(range(0, Component.shape[1]),
                               range(0, Component.shape[0]))
            Pos = np.empty(X.shape + (2,))
            Pos[:, :, 1] = X
            Pos[:, :, 0] = Y

            # initialize list of channels containing no non-degenerate objects
            Delete = []

            # for each color in rag
            for j in np.arange(1, np.max(RAG)+1):

                # get indices of cells to model in color 'j'
                Indices = np.nonzero(RAG == j)[0]

                # initialize count of component objects that were modeled
                Count = 0

                # for each nucleus in color
                for k in Indices:

                    # define x, y coordinates of nucleus
                    cY, cX = np.nonzero(Component == k+1)

                    # model each nucleus with gaussian and add to component 'j'
                    Mean, Cov = _gaussian_model(Response[Locations[i-1]],
                                                cX, cY)
                    try:
                        Model = mvn(Mean.flatten(), Cov.squeeze())
                    except:
                        continue

                    # increment counter
                    Count += 1

                    # add multivariate normal to channel 'j' of D
                    D[:, :, j] = np.maximum(D[:, :, j], Model.pdf(Pos))

                # add channel j to delete list of no objects were added
                if(Count == 0):
                    Delete.append(j)

            # add background probability
            D[:, :, 0] = Background

            # keep channels containing non-degenerate objects
            if len(Delete):
                Temp = np.zeros((Component.shape[0], Component.shape[1],
                                 D.shape[2]-len(Delete)), dtype=np.float)
                Channel = 0
                for j in np.arange(D.shape[2]):
                    if j not in Delete:
                        Temp[:, :, Channel] = D[:, :, j]
                        Channel += 1
                D = Temp

            # component contains non-degenerate objects - cut
            if(D.shape[2] > 1):

                # score probabilities
                for j in np.arange(D.shape[2]):
                    D[:, :, j] = -np.log(D[:, :, j] + np.finfo(np.float).eps)

                # formulate image-based gradient costs
                Patch = I[Locations[i-1]].astype(np.float)
                Horizontal = np.exp(-np.abs(Patch[:, 0:-1] - Patch[:, 1:]))
                Vertical = np.exp(-np.abs(Patch[0:-1, :] - Patch[1:, :]))

                # formulate label cost
                V = 1 - np.identity(D.shape[2])
                V = Smoothness * V

                # cut the graph and reshape the output
                Cut = gc.cut_grid_graph(D, V, Vertical, Horizontal,
                                        n_iter=-1, algorithm='swap')
                Cut = Cut.reshape(Component.shape[0],
                                  Component.shape[1]).astype(np.uint32)

                # split the labels that were grouped during graph coloring
                Cut = lb.SplitLabel(Cut)

                # capture number of objects in cut result
                Max = Cut.max()

                # update the values in the cut
                Cut[Cut > 0] = Cut[Cut > 0] + Total

                # embed the resulting cut into the output label image
                Refined[Components == i] = Cut[ComponentMask]

                # update object count
                Total = Total + Max

            else:  # component is degenerate - contains no viable objects

                Refined[Components == i] = 0

        else:  # single object component - no refinement necessary

            # increment object count
            Total += 1

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

    # zero weights below zero
    Weights[Weights < 0] = 0

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
