import imageio
from skimage import io, transform
from skimage.color import rgb2gray
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
import matplotlib.pyplot as plt
import torch as t
from pathlib import Path
import os

def readImage(img_dir, has_mask, plot=False):
    #file = Path(img_dir)
    if '.DS_Store' not in str(file):
        item = {}
        imgs = []
        for image in Path(img_dir + '/images').iterdir():
            img = io.imread(image) # (256, 256, 4) ~ this is an RGBA image 
            if plot:
                plt.imshow(img)
                plt.axis('off')
                plt.title('RGBA Image')
                plt.show()
            imgs.append(img)
        # verify there is only one input image
        assert len(imgs)==1
        if img.shape[2]>3: # check if RGB (3) or RGBA (4)
            assert(img[:,:,3]!=255).sum()==0 # Q: if every entry is 255 so it's safe to convert to RGB?
        img = img[:,:,:3] # convert RGBA to RGB
        if plot:
            plt.imshow(img)
            plt.axis('off')
            plt.title('RGB Image')
            plt.show()
        if has_mask:
            mask_files = list(Path( img_dir + '/masks').iterdir())
            masks = None
            for ii,mask in enumerate(mask_files):
                mask = io.imread(mask)
                assert (mask[(mask!=0)]==255).all() #??
                if masks is None:
                    H,W = mask.shape
                    masks = np.zeros((len(mask_files),H,W))
                masks[ii] = mask
            tmp_mask = masks.sum(0)
            if plot:
                plt.imshow(tmp_mask)
                plt.axis('off')
                plt.title('Before Mask:')
                plt.show()
            assert (tmp_mask[tmp_mask!=0] == 255).all()
            for ii,mask in enumerate(masks):
                masks[ii] = mask/255 * (ii+1)
            mask = masks.sum(0)
            if plot:
                plt.imshow(tmp_mask)
                plt.axis('off')
                plt.title('After Mask:')
                plt.show()
            # pylint: disable=E1101
            item['mask'] = t.from_numpy(mask)
            # pylint: enable=E1101
        item['name'] = str(file).split('/')[-1]
        # pylint: disable=E1101
        item['img'] = t.from_numpy(img)
        # pylint: enable=E1101
        return item

if __name__ == '__main__':
    #img_dir = "/Users/lg186018/Documents/github/2018_data_science_bowl/unet/data/stage1_train/5908488d940e846cc121c768758da9b1bd5b9922417e20c9101a4e254fa98af8"
    #fileItem = readImage(img_dir, True, True)
    dir = os.path.dirname(__file__)
    print dir
    img_dir = dir + "/data/stage1_train/5908488d940e846cc121c768758da9b1bd5b9922417e20c9101a4e254fa98af8"
    p = Path( img_dir + '/masks')
    print p
    for pp in p.iterdir():
        print "> "+ str(pp)
