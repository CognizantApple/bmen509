#! Python 3
import cv2
import numpy as np
from math import sqrt, exp
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
# The set of scales at which to run the vesselness scoring
vesselness_scales = [1.5,2.1,3.0,3.4]
vesselness_coefficients = [1.0, 1.1, 1.3, 1.5]
# beta is a threshold controlling the sensitivity to the blobness measure
beta = 0.80
# c is a threshold controlling the sensitivity to second-order structureness.

# Threshold on maximum brightness at which a pixel can be part of a vessel
brightness_threshold = 160

def compute_vesselness_multiscale(image, debug_vessel_scores=True, c=(150.0/255.0)):
    '''
    Run the vesselness computation at several different scales,
    and take the highest score!
    '''
    image = image.astype(np.double)
    vesselness_scores = []
    final_scores = np.zeros((image.shape[0], image.shape[1]))

    # Run through each scale and compute the vesselness score for the image at that scale
    for i in range(len(vesselness_scales)):
        scale_scores = np.clip(compute_vesselness(image, vesselness_scales[i], c) * vesselness_coefficients[i], 0.0, 1.0)
        # scale_scores = compute_vesselness(image, vesselness_scales[i], c)
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

def compute_vesselness(image, scale, c=(150.0/255.0)):
    '''
    Compute the vesselness at a specific scale.
    '''
    # find the hessian at a certain scale
    hessian = hessian_matrix(image, sigma=scale)
    hessian_eigenvalues = hessian_matrix_eigvals(hessian)

    # Blobness measure
    R_beta = hessian_eigenvalues[1] / hessian_eigenvalues[0]
    # Second-order structureness - Don't bother squaring it because we would just take the root later.
    so_structureness_squared = np.square(hessian_eigenvalues[1]) + np.square(hessian_eigenvalues[0])

    max_s = np.max(np.sqrt(so_structureness_squared))

    c = 0.105 * max_s

    # Vesselness scoring function in 2d credit to A. Frangi et al.
    vesselness_pt1 = np.exp(-( (np.square(R_beta)) / ( 2 * (np.square(beta)))))
    vesselness_pt2 = (1 - np.exp(-( (so_structureness_squared) / ( 2 * (c**2)))))
    vesselness = vesselness_pt1 * vesselness_pt2


    no_vessel_idx = np.logical_or(hessian_eigenvalues[0] <= 0, hessian_eigenvalues[1] <= 0, image >= brightness_threshold)
    vesselness[no_vessel_idx] = 0

    return vesselness


def compute_avg_grey_value(image, filter):
    '''
    Computes the average grey value in image after applying filter,
    where "image" the green channel of an RGB image, and "filter" is some binary
    image. Both images are assumed to be numpy arrays with the same dimensions.
    '''
    filtered_im = np.multiply(image, filter)
    total_grey_value = 0
    num_grey_values = 0

    # For each pixel in the filtered image...
    for x, y in np.ndindex(filtered_im.shape):
        # If the value is non-zero (passed the filter),
        # add it to the cumulative grey score.
        if(filtered_im[x,y]):
            total_grey_value += filtered_im[x,y]
            num_grey_values += 1

    avg_value = total_grey_value / num_grey_values

    return avg_value

def apply_vesselness_enhancement(image, scores):
    '''
    Performs brightness enhancement on an image based on a set of vesselness
    scores for the image, where "image" is an RGB image and "scores" is a
    set of vesselness scores. Both "image" and "scores" are assumed to
    be numpy arrays with the same dimensions.
    '''
    # Colorspace transformation from RGB to HSV for brightness
    # enhancement.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]

    # For each pixel in the hsv image...
    for x, y in np.ndindex(v_channel.shape):
        # Brightness scaling factor is 1 minus the vesselness score
        # (so that pixels with a higher vesselness score become darker)...
        factor = 1 - scores[x,y]
        # Apply the brightness scaling factor to the V-channel
        # of the HSV image.
        hsv_image[x,y,2] = v_channel[x,y] * factor

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def remove_small_vessels(image, min_vessel_size):
    '''
    Removes objects in "image" which are smaller than "min_vessel_size".
    Assumes that "image" is a binary numpy array.
    '''
    # Gets each object in the binary image (assumes 8-connectivity)...
    num_objects, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # Counts background as an object, so we remove it from the results...
    sizes = stats[1:, -1]
    num_objects = num_objects - 1

    # Remove objects smaller than min_vessel_size...
    output_image = np.zeros((output.shape))
    for i in range(0, num_objects):
        if sizes[i] >= min_vessel_size:
            output_image[output == i + 1] = 255

    return output_image