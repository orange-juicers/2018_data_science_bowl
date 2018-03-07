import imageio
from skimage import io, transform
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
import matplotlib.pyplot as plt

class Exp:
    def __init__(self, training_path="", testing_path=""):
        self.training_path = training_path
        self.testing_path = testing_path

    def analyze_image(self, im_path, plot=False):
        '''
        Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
        and dump it into a Pandas DataFrame.
        '''
        # Read in data and convert to grayscale
        im_id = im_path.parts[-3]
        im = io.imread(str(im_path))
        #im_gray = rgb2gray(im)
        im_gray = io.imread(str(im_path),as_grey=True)
        if plot:
            plt.figure(figsize=(10,4))

            plt.subplot(1,2,1)
            plt.imshow(im)
            plt.axis('off')
            plt.title('Original Image')
            
            plt.subplot(1,2,2)
            plt.imshow(im_gray, cmap='gray')
            plt.axis('off')
            plt.title('Grayscale Image')
            
            plt.tight_layout()
            plt.show()