import numpy as np
import math

def window_level_function(image, window, level):
    # As recommended, convert the input image to a double
    image = image.astype(np.double)

    # Assume that the min and max are supposed to be 0 and 255 here,
    # Rather than np.min(image[:]) and np.max(image[:])
    mi = 0.0
    ma = 255.0

    # Calculations for m and b given in the laboratory notes!
    m = (ma-mi) / window
    b = ma - ((ma - mi) / window) * (level + (0.5 * window))

    # Loop through all the pixels, implement the window level function
    for (x,y), value in np.ndenumerate(image):
        if(image[x,y] < (level - (0.5 * window))):
            # Pixel intensity is below window threshold
            image[x,y] = 0
        elif(image[x,y] > (level + (0.5 * window))):
            # Pixel intensity is above window threshold
            image[x,y] = ma
        else:
            # Pixel intensity is in the window, so apply J(x,y) = m * image(x,y) + b
            image[x,y] = m * image[x,y] + b

    return image.astype(np.uint8)

def ExpandMask(input, iters):
    """
    Expands the True area in an array 'input'.

    Expansion occurs in the horizontal and vertical directions by one
    cell, and is repeated 'iters' times.
    """
    yLen,xLen = input.shape
    output = input.copy()
    for iter in range(iters):
      for y in range(yLen):
        for x in range(xLen):
          if (y > 0        and input[y-1,x]) or \
             (y < yLen - 1 and input[y+1,x]) or \
             (x > 0        and input[y,x-1]) or \
             (x < xLen - 1 and input[y,x+1]): output[y,x] = True
      input = output.copy()
    return output

def check_value_in_value_list(value, val_array):
    ''' Checks if the value is in the array of values! This is useful for seeing if
    an array is contained in a bigger list of arrays
    '''
    for val in val_array:
        if np.array_equal(value, val):
            return True
    return False


def closest_point(point, point_arr):
    '''Find the 2D point in point_arr
    That is closest to the arg point'''
    # compute the distance from the point to each
    # other point in either dimension
    diffs = point_arr - point
    # Use einsum for speed, but we're really just
    # squaring the distance between points in each
    # dimension and summing them up.
    min_dist = np.einsum('ij,ij->i', diffs, diffs)
    # Select the point at the shortest distance!
    return point_arr[np.argmin(min_dist)]

