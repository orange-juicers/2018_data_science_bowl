import imageio
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
import matplotlib.pyplot as plt

class Stephen:
    
    def __init__(self, training_path="", testing_path=""):
        self.training_path = training_path
        self.testing_path = testing_path
        
    def train(self, output_csv_path="stephen_train_output.csv"):
        df = self.analyze_list_of_images(list(self.training_path))
        df.to_csv(output_csv_path, index=None)
        
    def test(self, output_csv_path="stephen_test_output.csv"):
        df = self.analyze_list_of_images(list(self.testing_path))
        df.to_csv(output_csv_path, index=None)
    
    def analyze_image(self, im_path, plot=False):
        '''
        Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
        and dump it into a Pandas DataFrame.
        '''
        # Read in data and convert to grayscale
        im_id = im_path.parts[-3]
        im = imageio.imread(str(im_path))
        im_gray = rgb2gray(im)
    
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
            
        # Mask out background and extract connected objects
        thresh_val = threshold_otsu(im_gray)
        mask = np.where(im_gray > thresh_val, 1, 0)
        if np.sum(mask==0) < np.sum(mask==1):
            mask = np.where(mask, 0, 1)    
            labels, nlabels = ndimage.label(mask)
        labels, nlabels = ndimage.label(mask)
        
        if plot:
            plt.figure(figsize=(10,4))

            plt.subplot(1,2,1)
            im_pixels = im_gray.flatten()
            plt.hist(im_pixels,bins=50)
            plt.vlines(thresh_val, 0, 100000, linestyle='--')
            plt.ylim([0,50000])
            plt.title('Grayscale Histogram')
            
            plt.subplot(1,2,2)
            mask_for_display = np.where(mask, mask, np.nan)
            plt.imshow(im_gray, cmap='gray')
            plt.imshow(mask_for_display, cmap='rainbow', alpha=0.5)
            plt.axis('off')
            plt.title('Image w/ Mask')
            plt.show()
    
        # detect edge using sobel function
        #im_gray_sx = ndimage.sobel(im_gray, axis = 0, mode = 'constant')
        #im_gray_sy = ndimage.sobel(im_gray, axis = 1, mode = 'constant')
        #im_gray_sob = np.hypot(im_gray_sx, im_gray_sy)
        #if plot:
        #    plt.imshow(im_gray_sob)
        #    plt.show()

        # Loop through labels and add each to a DataFrame
        im_df = pd.DataFrame()
        for label_num in range(1, nlabels+1):
            label_mask = np.where(labels == label_num, 1, 0)
            if label_mask.flatten().sum() > 10:
                rle = self.rle_encoding(label_mask)
                s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
                im_df = im_df.append(s, ignore_index=True)
    
        return im_df


    def analyze_list_of_images(self, im_path_list):
        '''
        Takes a list of image paths (pathlib.Path objects), analyzes each,
        and returns a submission-ready DataFrame.'''
        all_df = pd.DataFrame()
        for im_path in im_path_list:
            im_df = self.analyze_image(im_path)
            all_df = all_df.append(im_df, ignore_index=True)
        
        return all_df
    
    def rle_encoding(self, x):
        '''
        x: numpy array of shape (height, width), 1 - mask, 0 - background
        Returns run length as list
        '''
        dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1): run_lengths.extend((b+1, 0))
            run_lengths[-1] += 1
            prev = b
        return " ".join([str(i) for i in run_lengths])
