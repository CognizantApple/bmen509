#! Python 3
import numpy as np
from math import sqrt, exp
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
# The set of scales at which to run the vesselness scoring
#vesselness_scales = [0.1,0.2,0.3,0.5,0.8,1.3,2.1,4,5,6,7,8,9,10,11,12,13,14,15,16]
#vesselness_scales = [0.8,1.3,1.6,2.0,2.4,2.8,3.3,3.8,4.3,4.8,5.3,5.8,6.3,6.8,7.3]
vesselness_scales = [2.2,2.6,3.0,3.4]
# beta is a threshold controlling the sensitivity to the blobness measure
# TODO: goof around with this threshold and see if it helps.
beta = 0.75
# c is a threshold controlling the sensitivity to second-order structureness.
# for now this is just a rough guess, but there may actually be a much better choice for c ;)
c = (130.0/255.0)
# Threshold on maximum brightness at which a pixel can be part of a vessel
brightness_threshold = 130

#Show the intermediate images with vesselness at different scales!
debug_vessel_scores = True

def compute_vesselness_multiscale(image):
    '''
    Run the vesselness computation at several different scales,
    and take the highest score!
    '''
    image = image.astype(np.double)
    vesselness_scores = []
    final_scores = np.zeros((image.shape[0], image.shape[1]))

    # Run through each scale and compute the vesselness score for the image at that scale
    for scale in vesselness_scales:
        scale_scores = compute_vesselness(image, scale)
        vesselness_scores.append(scale_scores)
        final_scores = np.maximum(scale_scores, final_scores)

    # If you like, show an image of what the scores look like at all the different scales
    if(debug_vessel_scores):
        plt.subplots(2,2)
        i = 1
        for score in vesselness_scores:
            plt.subplot(2,2,i)
            plt.imshow(score)
            plt.title('Vesseness - Scale = ' + str(vesselness_scales[i-1]))
            plt.colorbar()
            i += 1

    plt.show()
    return final_scores

def compute_vesselness(image, scale):
    '''
    Compute the vesselness at a specific scale.
    '''
    # find the hessian at a certain scale
    hessian = hessian_matrix(image, sigma=scale)
    hessian_eigenvalues = hessian_matrix_eigvals(hessian)

    scores = np.zeros((image.shape[0], image.shape[1]))

    # Run through each pixel in the image
    for x, y in np.ndindex(image.shape):
        # in hessian_eigenvalues, the values are ordered from largest to smallest,
        # but we want the first eigenvalue to be the smallest.
        eigenvalue_xy_1 = hessian_eigenvalues[1,x,y]
        eigenvalue_xy_2 = hessian_eigenvalues[0,x,y]

        vesselness = 0.0
        if(eigenvalue_xy_2 > 0 and image[x,y] < brightness_threshold):

            # Blobness measure
            R_beta = eigenvalue_xy_1 / eigenvalue_xy_2
            # Second-order structureness
            so_structureness = sqrt(eigenvalue_xy_1**2 + eigenvalue_xy_2**2)

            # Vesselness scoring function in 2d credit to A. Frangi et al.
            vesselness = exp(-( (R_beta**2) / ( 2 * (beta**2)))) * \
                         (1 - exp(-( (so_structureness**2) / ( 2 * (c**2)))))

        scores[x,y] = vesselness

    return scores