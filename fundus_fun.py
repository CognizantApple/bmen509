#! python 3
from PIL import Image
import os
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import traceback, sys

'''
This is a super fun program for finding vessels in an image of a fundus!
The original image is enhanced and segmented, and scored against a hand-
segmented image.
'''

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
        self.raw_image = raw_image      # Just the raw image right off the disk
        self.label_image = label_image  # The corresponding hand-labelled image.

        self.vesselness_score = None    # single-channel 'image' containing vesselness scores
        self.enhanced_image = None      # RGB-channel enhanced version of the raw image.
        self.segmented_image = None     # Binary image containing segmentation information
        self.segment_score_image = None # RGB-channel colourized version of the scored segmented image.
                                        # true positives are white, true negatives black, false positives blue,
                                        # false negatives red.

class Fundus_Fun_App:
    def __init__(self):

        # As a basic first step, load up the image and label
        image_file = default_image
        label_file = default_label
        images_folder = default_images_folder
        labels_folder = default_labels_folder
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, label_file)

        image = self.load_image(image_path)
        label = self.load_image(label_path)

        image_container = Image_Container(image, label)

        # Simply display using matplotlib.
        plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.imshow(image_container.raw_image)
        plt.title(image_file + ' - Image')
        plt.subplot(2,1,2)
        plt.imshow(image_container.label_image)
        plt.title(label_file + ' - Label')
        plt.show()

    def load_image(self, image_filepath=None):
        if(image_filepath is None):
            # Allow the user to directly pick some images if they don't give an argument
            image_filepath = askopenfilename(filetypes = (("ppm images","*.ppm"),("all files","*.*")))
            if not os.path.exists(image_filepath):
                raise Exception('This file does not exist: {}'.format(image_filepath))

        image = Image.open(image_filepath)
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

