import histomicstk as htk
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

from NucleiCuts.nuclear_cut import nuclear_cut

W = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])
# Directory = '/Users/lcoop22/Desktop/Detection/'
Directory = '/home/cdeepakroy/Downloads/'

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

# extract nuclear channel
Nuclei = UN1.Stains[::2, ::2, 0].astype(dtype=np.uint8)

# cut
Label, Refined = nuclear_cut(Nuclei, Sigma=25, SigmaMin=4*(2**0.5),
                             SigmaMax=7*(2**0.5), r=10,
                             MinArea=20, MinWidth=5, Background=1e-4,
                             Smoothness=1e-4)

plt.figure()
plt.imshow(htk.EmbedBounds(I1[::2, ::2, 0],
                           htk.LabelPerimeter(Label) > 0,
                           Color=[255, 0, 0]))
plt.figure()
plt.imshow(htk.EmbedBounds(I1[::2, ::2, 0],
                           htk.LabelPerimeter(Refined) > 0,
                           Color=[255, 0, 0]))
