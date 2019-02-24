#! python 3
import cv2
import os
from vesselness_process import *
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import traceback, sys
import numpy as np

'''
This is a super fun program for finding vessels in an image of a fundus!
The original image is enhanced and segmented, and scored against a hand-
segmented image.
'''

# Should we show the green channel of the image during processing?
debug_original_img = False
debug_green_img = False
debug_vesselness_score = True
debug_segmented_image = True
debug_segment_score_image = True

root_dir = cwd = os.getcwd()
default_images_folder = os.path.join(root_dir, "images")
default_labels_folder = os.path.join(root_dir, "labels")
default_image = "im0001.ppm"
default_label = "im0001.vk.ppm"

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
        self.vesselness_threshold = 0.64 # Threshold at which we call the vesselness score a vessel!

        self.true_pos_colour  = [255,255,255] # white
        self.true_neg_colour  = [0  ,0  ,0  ] # black
        self.false_pos_colour = [0  ,0  ,255] # blue
        self.false_neg_colour = [255,0  ,0  ] # red

    def find_vesselness_image(self):
        '''
        Compute the vesselness score at each point in the image!
        '''

        # Extract only the green channel of the raw image for processing
        self.green_image = self.raw_image[:,:,1]
        if(debug_green_img):
            plt.figure()
            plt.imshow(self.green_image)
            plt.title('Green Channel only')
            plt.show()

        # Compute the vesselness of the image!
        self.vesselness_score = compute_vesselness_multiscale(self.green_image)

        if(debug_vesselness_score):
            plt.imshow(self.vesselness_score)
            plt.title('Vesselness score - overall')
            plt.colorbar()
            plt.show()

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
        if(debug_segmented_image):
            plt.imshow(self.segmented_image, cmap='gray')
            plt.title('Vesselness segmentation')
            plt.show()

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

        if(debug_segment_score_image):
            plt.imshow(self.segment_score_image)
            plt.title(('Segmentation score : Accuracy = {0:.4f} : Sensitivity = {1:.4f} : ' + \
                       'Specificity = {2:.4f}').format(self.accuracy, self.sensitivity, self.specificity))
            plt.show()

    def generate_enhanced_image(self):
        '''
        Use the vesselness scores and the original image
        to enhance the areas where there are vessels!
        '''
        self.enhanced_image




class Fundus_Fun_App:
    def __init__(self):

        # As a basic first step, load up the image and label
        image_file = default_image
        label_file = default_label
        images_folder = default_images_folder
        labels_folder = default_labels_folder
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, label_file)

        image = self.load_image(image_path, True)
        label = self.load_image(label_path, False)

        image_container = Image_Container(image, label)

        # Simply display using matplotlib.
        if(debug_original_img):
            plt.subplots(2,1)
            plt.subplot(2,1,1)
            plt.imshow(image_container.raw_image)
            plt.title(image_file + ' - Image')
            plt.subplot(2,1,2)
            plt.imshow(image_container.label_image)
            plt.title(label_file + ' - Label')
            plt.show()

        image_container.find_vesselness_image()
        image_container.generate_segmented_image()
        image_container.score_segmented_image()

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


if __name__ == '__main__':
    try:
        application = Fundus_Fun_App()
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        # print traceback
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        # print exception
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                limit=2, file=sys.stdout)

