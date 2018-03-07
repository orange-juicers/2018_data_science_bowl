import os
from pathlib import Path
from PIL import Image
from skimage import io
import numpy as np
import torch as t
from tqdm import tqdm


def process(file_path, has_mask=True):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in tqdm(files):
        if '.DS_Store' in str(file):
            continue
        item = {}
        imgs = []
        for image in (file/'images').iterdir():
            img = io.imread(image)
            imgs.append(img)
        assert len(imgs)==1
        if img.shape[2]>3:
            assert(img[:,:,3]!=255).sum()==0
        img = img[:,:,:3]

        if has_mask:
            mask_files = list((file/'masks').iterdir())
            masks = None
            for ii,mask in enumerate(mask_files):
                mask = io.imread(mask)
                assert (mask[(mask!=0)]==255).all()
                if masks is None:
                    H,W = mask.shape
                    masks = np.zeros((len(mask_files),H,W))
                masks[ii] = mask
            tmp_mask = masks.sum(0)
            assert (tmp_mask[tmp_mask!=0] == 255).all()
            for ii,mask in enumerate(masks):
                masks[ii] = mask/255 * (ii+1)
            mask = masks.sum(0)
            # pylint: disable=E1101
            item['mask'] = t.from_numpy(mask)
            # pylint: enable=E1101
        item['name'] = str(file).split('/')[-1]
        # pylint: disable=E1101
        item['img'] = t.from_numpy(img)
        # pylint: enable=E1101
        datas.append(item)
    return datas
