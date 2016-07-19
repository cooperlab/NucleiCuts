NucleiSegmentation Application
=============================================

#### Overview:

Segments nuclei using a graph-cut based algorithm developed by
Al-Kofahi et. al.

#### Command line usage:

```
python NucleiSegmentation.py [-h] [-V] [--xml] [--binary_cut_sigma <double>]
                             [--local_max_search_radius <double>]
                             [--max_radius <double>]
                             [--min_nucleus_area <double>]
                             [--min_nucleus_diameter <double>]
                             [--min_radius <double>]
                             [--multiwaycut_background_cost <double>]
                             [--multiwaycut_smoothness <double>]
                             [--stain_1 {hematoxylin,eosin,dab}]
                             [--stain_2 {hematoxylin,eosin,dab}]
                             [--stain_3 {hematoxylin,eosin,dab,null}]
                             inputImageFile outputNucleiMaskFile

positional arguments:
  inputImageFile        Input image to be deconvolved (type: image)
  outputNucleiMaskFile  Output nuclei segmentation label mask (type: image)

optional arguments:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  --xml                 Produce xml description of command line arguments
  --binary_cut_sigma <double>
                        Sigma used in the pair-wise potential of the graph-cut
                        based foreground-background segmentation (default:
                        25.0)
  --local_max_search_radius <double>
                        Local max search radius used for detection seed points
                        in nuclei (default: 10.0)
  --max_radius <double>
                        Maximum nuclear radius (used to set max sigma of the
                        multiscale LoG filter) (default: 7.0)
  --min_nucleus_area <double>
                        Minimum area that each nucleus should have (default:
                        20.0)
  --min_nucleus_diameter <double>
                        Minimum diameter each nucleus should have (default:
                        5.0)
  --min_radius <double>
                        Minimum nuclear radius (used to set min sigma of the
                        multiscale LoG filter) (default: 4.0)
  --multiwaycut_background_cost <double>
                        Cost of assigning background pixels to the background
                        label in multi-way cut refinement (default: 0.0001)
  --multiwaycut_smoothness <double>
                        Weight of smoothness/pairwise energy term in the
                        multi-way cut refinement step (default: 0.0001)
  --stain_1 {hematoxylin,eosin,dab}
                        Name of stain-1 (default: hematoxylin)
  --stain_2 {hematoxylin,eosin,dab}
                        Name of stain-2 (default: eosin)
  --stain_3 {hematoxylin,eosin,dab,null}
                        Name of stain-3 (default: null)
```
