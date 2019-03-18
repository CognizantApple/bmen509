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
    deltas = point_arr - point
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return point_arr[np.argmin(dist_2)]

# taking the negative of these numbers changes the up to down!
move_right_boundary     = 0.3927     # 22.5 degrees
move_up_right_boundary  = 1.1781     # 67.5 degrees
move_up_left_boundary   = 1.9635     # 112.5 degrees
move_left_boundary      = 2.7489     # 157.5 degrees
y_idx = 1
x_idx = 0
