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

root_dir = cwd = os.getcwd()
images_folder = os.path.join(root_dir, "images")
labels_folder = os.path.join(root_dir, "labels")
healthy_images_folder = os.path.join(root_dir, "healthyimages")
healthy_labels_folder = os.path.join(root_dir, "healthylabels")
saved_images_folder   = os.path.join(root_dir, "Saved")

preprocessed_images_folder = os.path.join(root_dir, "preprocessed")


class Fundus_Fun_App:
    def __init__(self):

        # As a basic first step, load up the image and label
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.healthy_images_folder = healthy_images_folder
        self.healthy_labels_folder = healthy_labels_folder

        self.container_list = []
        self.healthy_container_list = []

    def load_all_images(self):
        self.container_list = self.load_images(self.images_folder, self.labels_folder)
        self.healthy_container_list = self.load_images(self.healthy_images_folder, self.healthy_labels_folder)


    def load_images(self, images_folder, labels_folder):
        image_list = []
        image_name_list = []
        label_list = []
        for filename in glob.glob(images_folder + '/*.ppm'): #assuming ppm
            image_list.append(self.load_image(filename, True))
            image_name_list.append(os.path.splitext(os.path.basename(filename))[0])

        for filename in glob.glob(labels_folder + '/*.ppm'): #assuming ppm
            label_list.append(self.load_image(filename, False))

        if(len(image_list) != len(label_list)):
            print("Number of images and labels are not the same!")
            sys.exit(1)

        container_list = []

        for i in range(len(image_list)):
            container_list.append(Image_Container(image_list[i],label_list[i], image_name_list[i]))

        return container_list

    def process_images(self, container_list):
        for i in range(len(container_list)):
            # Simply display using matplotlib.
            if(debug_img):
                plt.subplots(2,1)
                plt.subplot(2,1,1)
                plt.imshow(container_list[i].raw_image)
                plt.title('Image - ' + str(i))
                plt.subplot(2,1,2)
                plt.imshow(container_list[i].label_image)
                plt.title('Label - ' + str(i))
                plt.show()

            container_list[i].preprocess_image()
            if(debug_img):
                plt.subplots(2,2)
                plt.subplot(2,2,1)
                plt.imshow(container_list[i].red_image, cmap='gray')
                plt.title('Image Red Channel')
                plt.subplot(2,2,2)
                plt.imshow(container_list[i].not_region_of_interest_mask, cmap='gray')
                plt.title('Outside region of interest mask')
                plt.subplot(2,2,3)
                plt.imshow(container_list[i].region_of_interest_edge_mask, cmap='gray')
                plt.title('Region of interest edge')
                plt.subplot(2,2,4)
                plt.imshow(container_list[i].preprocess_image, cmap='gray')
                plt.title('Preprocessed image')
                plt.show()

            container_list[i].find_vesselness_image()
            container_list[i].generate_segmented_image()
            container_list[i].score_segmented_image()
            if(debug_img):
                plt.subplots(2,2)
                plt.subplot(2,2,1)
                plt.imshow(container_list[i].raw_image)
                plt.title('Original Image')
                plt.subplot(2,2,2)
                plt.imshow(container_list[i].vesselness_score, cmap='gray')
                plt.title('Vesselness score - overall, cropped')
                plt.subplot(2,2,3)
                plt.imshow(container_list[i].segmented_image, cmap='gray')
                plt.title('Vesselness segmentation')
                plt.subplot(2,2,4)
                plt.imshow(container_list[i].segment_score_image)
                plt.title(('Segmentation score : Accuracy = {0:.4f} : Sensitivity = {1:.4f} : ' + \
                        'Specificity = {2:.4f}').format(container_list[i].accuracy, container_list[i].sensitivity, container_list[i].specificity))
                plt.show()

            container_list[i].generate_enhanced_image()
            if(debug_img):
                plt.subplots(1,3)
                plt.subplot(1,3,1)
                plt.imshow(container_list[i].raw_image)
                plt.title('Original Image')
                plt.subplot(1,3,2)
                plt.imshow(container_list[i].vesselness_score, cmap='gray')
                plt.title('Vesselness score - overall, cropped')
                plt.subplot(1,3,3)
                plt.imshow(container_list[i].enhanced_image)
                plt.title('Enhanced Image')
                plt.show()

            if(save_imgs):
                self.save_image_container_images(container_list[i], saved_images_folder)

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

    def save_image_container_images(self, image_container, directory):
        '''
        Save the images of interest from an image_container once it's been processed.
        '''
        if not(os.path.exists(directory)):
            os.makedirs(directory)

        # Save each image of interest
        scipy.misc.imsave(os.path.join(directory, image_container.image_name + '_vesselness.jpg'), image_container.vesselness_score)
        scipy.misc.imsave(os.path.join(directory, image_container.image_name + '_segmented.jpg'), image_container.segmented_image)
        scipy.misc.imsave(os.path.join(directory, image_container.image_name + '_segment_score.jpg'), image_container.segment_score_image)
        scipy.misc.imsave(os.path.join(directory, image_container.image_name + '_preprocessed.jpg'), image_container.preprocess_image)
        scipy.misc.imsave(os.path.join(directory, image_container.image_name + '_enhanced.jpg'), image_container.enhanced_image)

    def compute_accuracy_stats(self, container_list):
        '''
        Compute overall accuracy stats for the entire set of processed images.
        '''
        accuracies = []
        sensitivities = []
        specificities = []
        for i in range(len(container_list)):
            print(container_list[i].image_name + ":")
            print("accuracy: {}".format(container_list[i].accuracy))
            print("sensitivity: {}".format(container_list[i].sensitivity))
            print("specificity: {}".format(container_list[i].specificity))
            accuracies.append(container_list[i].accuracy)
            sensitivities.append(container_list[i].sensitivity)
            specificities.append(container_list[i].specificity)

        print('Average accuracy: {}'.format(np.mean(accuracies)))
        print('Median accuracy: {}'.format(np.median(accuracies)))
        print('Std accuracy: {}'.format(np.std(accuracies)))
        print('Min accuracy: {0} ({1})'.format(np.min(accuracies), container_list[np.argmin(accuracies)].image_name))
        print('Max accuracy: {0} ({1})'.format(np.max(accuracies), container_list[np.argmax(accuracies)].image_name))
        print('Average sensitivity: {}'.format(np.mean(sensitivities)))
        print('Median sensitivity: {}'.format(np.median(sensitivities)))
        print('Std sensitivity: {}'.format(np.std(sensitivities)))
        print('Min sensitivity: {0} ({1})'.format(np.min(sensitivities), container_list[np.argmin(sensitivities)].image_name))
        print('Max sensitivity: {0} ({1})'.format(np.max(sensitivities), container_list[np.argmax(sensitivities)].image_name))
        print('Average specificity: {}'.format(np.mean(specificities)))
        print('Median specificity: {}'.format(np.median(specificities)))
        print('Std specificity: {}'.format(np.std(specificities)))
        print('Min specificity: {0} ({1})'.format(np.min(specificities), container_list[np.argmin(specificities)].image_name))
        print('Max specificity: {0} ({1})'.format(np.max(specificities), container_list[np.argmax(specificities)].image_name))


class Image_Container:
    '''
    This is a class for holding an image and all the different
    forms it takes as a result of processing.
    '''

    def __init__(self, raw_image, label_image, image_name):
        self.image_name = image_name
        self.raw_image = raw_image       # Just the raw image right off the disk
        self.label_image = label_image.astype(np.uint8)  # The corresponding hand-labelled image.

        self.enhanced_image = None       # RGB-channel enhanced version of the raw image.
        self.segmented_image = None      # Binary image containing segmentation information
        self.segment_score_image = None  # RGB-channel colourized version of the scored segmented image.
                                         # true positives are white, true negatives black, false positives blue,
                                         # false negatives red.
        self.vesselness_score = None     # single-channel 'image' containing vesselness scores
        self.vesselness_threshold = 0.29 # Threshold at which we call the vesselness score a vessel!
        self.vesselness_min_size = 200   # Minimum size for a vessel (in pixels)
        self.accuracy = 0
        self.specificity = 0
        self.sensitivity = 0

        self.true_pos_colour  = [255,255,255] # white
        self.true_neg_colour  = [0  ,0  ,0  ] # black
        self.false_pos_colour = [0  ,0  ,255] # blue
        self.false_neg_colour = [255,0  ,0  ] # red

    def preprocess_image(self):
        ''' Three major steps. first, use the red channel to create a mask for the area of interest.
        Then, use the red mask to change the outside of the green channel image to be the average colour
        in the region of interest.
        Finally, use a window and level function to remove white blotches that are misinterpreted as vessels.
        '''

        ''' STEP 2 - Use red channel to find ROI'''
        # Extract the red component of the image
        self.red_image = self.raw_image[:,:,0]
        self.green_image = self.raw_image[:,:,1]

        # find the region of NOT interest - white splotches and the black background!
        not_region_of_interest_idx = self.red_image < 39
        self.not_region_of_interest_mask = np.zeros(self.red_image.shape)
        self.not_region_of_interest_mask[not_region_of_interest_idx] = 255

        ''' STEP 3 - Find the edge of the ROI'''
        # Expand the region of non-interest by one pixel
        self.not_region_of_interest_mask = ExpandMask(self.not_region_of_interest_mask, 4)
        not_region_of_interest_idx = self.not_region_of_interest_mask == 1.0

        # Create a thin layer of pixels, which will be the boundary of the layer of interest.
        not_region_of_interest_mask_expanded = ExpandMask(self.not_region_of_interest_mask, 1)

        region_of_interest_edge = not_region_of_interest_mask_expanded - self.not_region_of_interest_mask
        self.region_of_interest_edge_mask = self.green_image.copy()
        self.region_of_interest_edge_mask[region_of_interest_edge == 1.0] = 255

        # We have our masks, at this point jump out if the preprocessed file already exists.
        processed_file = preprocessed_images_folder + '/' + self.image_name + '.ppm'
        if os.path.exists(processed_file): #assuming ppm
            self.preprocess_image = cv2.imread(processed_file, 0)
            return

        self.preprocess_image = self.green_image.copy()

        ''' STEP 4 - Blend periphery of image into ROI'''
        # For each pixel outside the region of interest, colour it to be the same as the
        # closest pixel inside the area of interest
        region_of_interest_list = np.asarray(np.argwhere(region_of_interest_edge == 1.0))

        for pixel in np.argwhere(not_region_of_interest_idx):
            pix = pixel
            pix = closest_point(pix, region_of_interest_list)
            self.preprocess_image[pixel[0], pixel[1]] = self.preprocess_image[pix[0], pix[1]]

        # Save the preprocessed image so we have the option of re-using it
        scipy.misc.imsave(processed_file, self.preprocess_image)

    def find_vesselness_image(self):
        ''' Compute the vesselness score at each point in the image!
        '''

        # Extract only the green channel of the raw image for processing
        if(self.preprocess_image is None):
            self.preprocess_image = self.self.green_image = self.raw_image[:,:,1]

        # Compute the vesselness of the image!
        self.vesselness_score = compute_vesselness_multiscale(self.preprocess_image, self.not_region_of_interest_mask, debug_img)

    def generate_segmented_image(self):
        '''
        Uses a threshold and the vesselness_score image to create
        a segmentation
        '''
        self.segmented_image = np.copy(self.vesselness_score)
        above_threshold_indices = self.segmented_image >= self.vesselness_threshold
        below_threshold_indices = self.segmented_image < self.vesselness_threshold
        self.segmented_image[above_threshold_indices] = 255 # Set vessels to white
        self.segmented_image[below_threshold_indices] = 0   # Non-vessels to black

        # Make sure the segmented image is using integer values
        self.segmented_image = self.segmented_image.astype(np.uint8)

        # Remove objects (specks) too small to be vessels from the segmentation
        self.segmented_image = remove_small_vessels(self.segmented_image, self.vesselness_min_size)

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
        self.enhanced_image = apply_vesselness_enhancement(self.raw_image, self.vesselness_score)

if __name__ == '__main__':
    application = Fundus_Fun_App()
    application.load_all_images()
    application.process_images(application.healthy_container_list)
    application.compute_accuracy_stats(application.healthy_container_list)
    application.process_images(application.container_list)
    application.compute_accuracy_stats(application.container_list)
