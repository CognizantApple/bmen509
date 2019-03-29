#! python 3
import cv2
import os
from vesselness_process import *
from process_utils import *
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import traceback, sys
import numpy as np
import scipy.misc
import glob

'''
This is a super fun program for finding vessels in an image of a fundus!
The original image is enhanced and segmented, and scored against a hand-
segmented image.
'''

# Should we show images during processing?
debug_img = False
save_imgs = True
recompute_c = False # Call to recompute C for the set of images being used.
                    # If false, a previously computed value of C is used.

root_dir = cwd = os.getcwd()
images_folder = os.path.join(root_dir, "images")
labels_folder = os.path.join(root_dir, "labels")
healthy_images_folder = os.path.join(root_dir, "healthyimages")
healthy_labels_folder = os.path.join(root_dir, "healthylabels")
saved_images_folder   = os.path.join(root_dir, "Saved")

preprocessed_images_folder = os.path.join(root_dir, "preprocessed")

def gaussian_kernel(sigma, kernel_size):
    # Make sure kernel_size is at least 1, and odd.
    if(kernel_size < 1 or kernel_size % 2 != 1):
        os.sys.exit('Invalid kernel size: {}'.format(kernel_size))

    # Start the kernel as an array of zeros
    kernel = np.zeros((kernel_size, kernel_size))

    # The centers of the kernel, ux and uy:
    ux = (kernel_size - 1) / 2
    uy = ux

    # Iterate over the whole kernel
    for x in range(0, kernel_size):
        for y in range(0, kernel_size):
            # Apply the formula given above! Figure out things piece by piece to make it more readable.
            exponent = -(((x-ux)**2 + (y-uy)**2)/(2*(sigma**2)))
            fraction_denominator = (2*np.pi)**(0.5)
            fraction_denominator = fraction_denominator * sigma
            kernel[x,y] = (1 / fraction_denominator) * np.exp(exponent)
    return kernel

class Kernel_Fun_Times:
    def __init__(self):

        self.size = 101

    def load_all_images(self):
        self.container_list = self.load_images(self.images_folder, self.labels_folder)
        self.healthy_container_list = self.load_images(self.healthy_images_folder, self.healthy_labels_folder)


    def make_kernels(self):
        self.base_kernel = gaussian_kernel(20, self.size)
        self.x2_kernel = self.base_kernel.copy()
        self.x2_kernel = np.diff(np.diff(self.x2_kernel, axis=1), axis=1)
        self.y2_kernel = np.zeros((self.size, self.size))
        self.y2_kernel = self.base_kernel.copy()
        self.y2_kernel = np.diff(np.diff(self.y2_kernel, axis=0), axis=0)
        self.xy_kernel = self.base_kernel.copy()
        self.xy_kernel = np.diff(np.diff(self.xy_kernel, axis=1), axis=0)

    def show_kernels(self):
        plt.subplots(2,2)
        plt.subplot(2,2,1)
        plt.imshow(self.base_kernel)
        plt.title('Gaussian Kernel')
        plt.subplot(2,2,2)
        plt.imshow(self.x2_kernel)
        plt.title('Gaussian Kernel, 2nd derivative in X')
        plt.subplot(2,2,3)
        plt.imshow(self.y2_kernel)
        plt.title('Gaussian Kernel, 2nd derivative in Y')
        plt.subplot(2,2,4)
        plt.imshow(self.xy_kernel)
        plt.title('Gaussian Kernel, derivative in X and Y')
        plt.show()

if __name__ == '__main__':
    application = Kernel_Fun_Times()
    application.make_kernels()
    application.show_kernels()
