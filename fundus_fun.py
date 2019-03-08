#! python 3
import cv2
import os
from vesselness_process import *
from process_utils import *
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import traceback, sys
import numpy as np
import glob

'''
This is a super fun program for finding vessels in an image of a fundus!
The original image is enhanced and segmented, and scored against a hand-
segmented image.
'''

# Should we show images during processing?
debug_img = True

root_dir = cwd = os.getcwd()
default_images_folder = os.path.join(root_dir, "images")
default_labels_folder = os.path.join(root_dir, "labels")
default_image = "im0002.ppm"
default_label = "im0002.vk.ppm"

class Fundus_Fun_App:
    def __init__(self):

        # As a basic first step, load up the image and label
        image_file = default_image
        label_file = default_label
        images_folder = default_images_folder
        labels_folder = default_labels_folder
        # image_path = os.path.join(images_folder, image_file)
        # label_path = os.path.join(labels_folder, label_file)

        image_list = []
        label_list = []
        container_list = []
        for filename in glob.glob(images_folder + '/*.ppm'): #assuming ppm
            image_list.append(self.load_image(filename, True))

        for filename in glob.glob(labels_folder + '/*.ppm'): #assuming ppm
            label_list.append(self.load_image(filename, False))

        if(len(image_list) != len(label_list)):
            print("Number of images and labels are not the same!")
            sys.exit(1)

        for i in range(len(image_list)):
            container_list.append(Image_Container(image_list[i],label_list[i]))

            # Simply display using matplotlib.
            if(debug_img):
                plt.subplots(2,1)
                plt.subplot(2,1,1)
                plt.imshow(container_list[-1].raw_image)
                plt.title('Image - ' + str(i))
                plt.subplot(2,1,2)
                plt.imshow(container_list[-1].label_image)
                plt.title('Label - ' + str(i))
                plt.show()

            container_list[-1].preprocess_image()
            if(debug_img):
                plt.subplots(2,2)
                plt.subplot(2,2,1)
                plt.imshow(container_list[-1].not_region_of_interest_mask, cmap='gray')
                plt.title('Outside region of interest mask')
                plt.subplot(2,2,2)
                plt.imshow(container_list[-1].region_of_interest_edge_mask, cmap='gray')
                plt.title('Region of interest edge')
                plt.subplot(2,2,3)
                plt.imshow(container_list[-1].preprocess_image, cmap='gray')
                plt.title('Preprocessed image')
                plt.show()

            container_list[-1].find_vesselness_image()
            container_list[-1].generate_segmented_image()
            container_list[-1].score_segmented_image()
            if(debug_img):
                plt.subplots(2,2)
                plt.subplot(2,2,1)
                plt.imshow(container_list[-1].raw_image)
                plt.title('Vesselness score - overall')
                plt.subplot(2,2,2)
                plt.imshow(container_list[-1].vesselness_score)
                plt.title('Vesselness score - overall, cropped')
                plt.subplot(2,2,3)
                plt.imshow(container_list[-1].segmented_image, cmap='gray')
                plt.title('Vesselness segmentation')
                plt.subplot(2,2,4)
                plt.imshow(container_list[-1].segment_score_image)
                plt.title(('Segmentation score : Accuracy = {0:.4f} : Sensitivity = {1:.4f} : ' + \
                        'Specificity = {2:.4f}').format(container_list[-1].accuracy, container_list[-1].sensitivity, container_list[-1].specificity))
                plt.show()

    def load_image(self, image_filepath=None, convert_from_bgr=False):
        if(image_filepath is None):
            # Allow the user to directly pick some images if they don't give an argument
            image_filepath = askopenfilename(filetypes = (("ppm images","*.ppm"),("all files","*.*")))
            if not os.path.exists(image_filepath):
                raise Exception('This file does not exist: {}'.format(image_filepath))

        image = cv2.imread(image_filepath)
        if(convert_from_bgr):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class Image_Container:
    '''
    This is a class for holding an image and all the different
    forms it takes as a result of processing. The actual processing
    is done by other functions or classes.
    '''

    def __init__(self, raw_image, label_image):
        self.raw_image = raw_image       # Just the raw image right off the disk
        self.label_image = label_image.astype(np.uint8)  # The corresponding hand-labelled image.

        self.enhanced_image = None       # RGB-channel enhanced version of the raw image.
        self.segmented_image = None      # Binary image containing segmentation information
        self.segment_score_image = None  # RGB-channel colourized version of the scored segmented image.
                                         # true positives are white, true negatives black, false positives blue,
                                         # false negatives red.
        self.vesselness_score = None     # single-channel 'image' containing vesselness scores
        self.vesselness_threshold = 0.45 # Threshold at which we call the vesselness score a vessel!

        self.true_pos_colour  = [255,255,255] # white
        self.true_neg_colour  = [0  ,0  ,0  ] # black
        self.false_pos_colour = [0  ,0  ,255] # blue
        self.false_neg_colour = [255,0  ,0  ] # red

    def preprocess_image(self):
        ''' Two major steps. first, use the red channel to create a mask for the area of interest.
        Then, use the red mask to change the outside of the green channel image to be the average colour
        in the region of interest.
        Finally, use a window and level function to remove white blotches that are misinterpreted as vessels.
        '''
        # Extract the red component of the image
        self.red_image = self.raw_image[:,:,0]
        self.green_image = self.raw_image[:,:,1]

        # find the region of NOT interest - white splotches and the black background!
        not_region_of_interest_idx = self.red_image < 39
        self.not_region_of_interest_mask = np.zeros(self.red_image.shape)
        self.not_region_of_interest_mask[not_region_of_interest_idx] = 255

        self.not_region_of_interest_mask = ExpandMask(self.not_region_of_interest_mask, 4)
        not_region_of_interest_idx = self.not_region_of_interest_mask == 1.0


        # Create a thin layer of pixels, which will be the boundary of the layer of interest.
        not_region_of_interest_mask_expanded = ExpandMask(self.not_region_of_interest_mask, 1)

        region_of_interest_edge = not_region_of_interest_mask_expanded - self.not_region_of_interest_mask
        self.region_of_interest_edge_mask = self.green_image.copy()
        self.region_of_interest_edge_mask[region_of_interest_edge == 1.0] = 255

        self.preprocess_image = self.green_image.copy()

        # For each pixel outside the region of interest, colour it to be the same as the
        # closest pixel inside the area of interest
        region_of_interest_list = np.asarray(np.argwhere(region_of_interest_edge == 1.0))

        for pixel in np.argwhere(not_region_of_interest_idx):
            pix = pixel
            pix = closest_point(pix, region_of_interest_list)
            self.preprocess_image[pixel[0], pixel[1]] = self.preprocess_image[pix[0], pix[1]]

        self.preprocess_image = window_level_function(self.preprocess_image, 180, 120)

    def find_vesselness_image(self):
        ''' Compute the vesselness score at each point in the image!
        '''

        # Extract only the green channel of the raw image for processing
        if(self.preprocess_image is None):
            self.preprocess_image = self.self.green_image = self.raw_image[:,:,1]

        # Compute the vesselness of the image!
        self.vesselness_score = compute_vesselness_multiscale(self.preprocess_image)

        # Exclude vessels outside the area of interest
        self.vesselness_score[self.not_region_of_interest_mask == 1.0] = 0

    def generate_segmented_image(self):
        '''
        Uses a threshold and the vesselness_score image to create
        a segmentation
        '''
        self.segmented_image = np.copy(self.vesselness_score)
        above_threshold_indices = self.segmented_image >= self.vesselness_threshold
        below_threshold_indices = self.segmented_image < self.vesselness_threshold
        self.segmented_image[above_threshold_indices] = 255
        self.segmented_image[below_threshold_indices] = 0

        self.segmented_image = self.segmented_image.astype(np.uint8)

    def score_segmented_image(self):
        '''
        Uses the segmented image and the ground truth image
        to generate scores for accuracy, specificity, and sensitivity.
        Then make a pretty image visualizing those scores!
        '''
        # Create a blank RGB image
        self.segment_score_image = np.zeros((self.segmented_image.shape[0], \
                                   self.segmented_image.shape[1],3), np.uint8)
        label = cv2.cvtColor(self.label_image, cv2.COLOR_RGB2GRAY)
        self.false_pos = 0
        self.false_neg = 0
        self.true_pos  = 0
        self.true_neg  = 0

        # Run through each pixel and make a note of true positives,
        # false positives, false negatives, and true negatives.
        for x, y in np.ndindex(self.segmented_image.shape):
            if(self.segmented_image[x,y] == 255 and label[x,y] == 255 ):
                self.true_pos = self.true_pos + 1
                self.segment_score_image[x,y] = self.true_pos_colour

            elif(self.segmented_image[x,y] == 0 and label[x,y] == 0 ):
                self.true_neg = self.true_neg + 1
                self.segment_score_image[x,y] = self.true_neg_colour

            elif(self.segmented_image[x,y] == 255 and label[x,y] == 0 ):
                self.false_pos = self.false_pos + 1
                self.segment_score_image[x,y] = self.false_pos_colour

            elif(self.segmented_image[x,y] == 0 and label[x,y] == 255 ):
                self.false_neg = self.false_neg + 1
                self.segment_score_image[x,y] = self.false_neg_colour

        self.accuracy    = (self.true_neg + self.true_pos) / \
                           (self.true_neg + self.true_pos + self.false_neg + self.false_pos)
        self.sensitivity = (self.true_pos) / (self.true_pos + self.false_neg)
        self.specificity = (self.true_neg) / (self.true_neg + self.false_pos)

    def generate_enhanced_image(self):
        '''
        Use the vesselness scores and the original image
        to enhance the areas where there are vessels!
        '''
        self.enhanced_image



if __name__ == '__main__':
    application = Fundus_Fun_App()
    # try:
    #     application = Fundus_Fun_App()
    # except Exception:
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     # print traceback
    #     traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
    #     # print exception
    #     traceback.print_exception(exc_type, exc_value, exc_traceback,
    #                             limit=2, file=sys.stdout)
