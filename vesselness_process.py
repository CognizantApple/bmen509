#! Python 3
import cv2
import numpy as np
from math import sqrt, exp
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
# The set of scales at which to run the vesselness scoring
vesselness_scales = [1.5,2.1,3.0,3.4]
vesselness_coefficients = [1.0, 1.2, 2.5, 3.5]
# beta is a threshold controlling the sensitivity to the blobness measure
beta = 0.80
# c is a threshold controlling the sensitivity to second-order structureness.
# for now this is just a rough guess, but there may actually be a much better choice for c ;)
# c = (150.0/255.0)
# Threshold on maximum brightness at which a pixel can be part of a vessel
brightness_threshold = 160

def compute_vesselness_multiscale(image, debug_vessel_scores=False, c=(150.0/255.0)):
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