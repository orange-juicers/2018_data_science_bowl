from os import listdir, makedirs
# from os import remove
from os.path import isfile, join, exists
# from os.path import isdir
from matplotlib.pyplot import subplot, imshow, clf
# from matplotlib.pyplot import figure, plot
from matplotlib.colors import rgb_to_hsv
import matplotlib.image as img
import numpy as np
import random
from shutil import copyfile
# from skimage.transform import resize
# from skimage.morphology import label


training_path = "../data/stage1_train"

def get_im_with_mask(directory, boundaries=True):

    image_dir = join(directory, "images")
    mask_dir = join(directory, "masks")

    image_names = listdir(image_dir)
    mask_names = listdir(mask_dir)

    image_path = join(image_dir, image_names[0])
    image = img.imread(image_path)

    mask_path = join(mask_dir, mask_names[0])
    mask = img.imread(mask_path)
    if boundaries:
        mask = outline_mask(mask)

    for i in range(1, len(mask_names)):
        mask_path = join(mask_dir, mask_names[i])
        if boundaries:
            mask += outline_mask(img.imread(mask_path))
        else:
            mask += img.imread(mask_path)
    return image[:, :, 0:3], mask


def plot_example():
    im, mask = get_example()
    plot_im_mask(im, mask)


def get_example():
    directory = training_path
    names = listdir(directory)
    index = random.randint(0, len(names)-1)
    im, mask = get_im_with_mask(join(directory, names[index]))
    return im, mask


def plot_im_mask(im, mask):
    clf()
    subplot(121)
    imshow(im)
    subplot(122)
    imshow(mask)


def pad_image(im, half):

    if len(im.shape) == 3:
        color = [0, 0, 0]
        im = np.stack(
            [np.lib.pad(im[:, :, c], half, mode='constant', constant_values=color[c]) for c in range(3)],
            axis=2
        )
    elif len(im.shape) == 2:
        im = np.lib.pad(im, half, mode="constant")
    else:
        print("Unwritten code!!")
    return im


def run_len_enc(input_matrix):
    num_cols, num_rows = input_matrix.shape
    input_array = np.reshape(input_matrix, [num_cols * num_rows], order='F')
    locations = np.where(input_array == 1)[0]
    run_len = []
    prev = -2
    for loc in locations:
        if loc > prev + 1:
            run_len.extend((loc + 1, 0))
        run_len[-1] += 1
        prev = loc
    return np.array(run_len)


def run_len_dec(input_array, num_cols, num_rows):
    out = np.zeros(num_cols * num_rows)
    n = len(input_array)
    for i in range(n):
        if i % 2 == 0:
            start = input_array[i]-1
            dist = input_array[i+1]
            out[start:start+dist] = 1
    out = np.reshape(out, [num_cols, num_rows], order='F')
    return out


def get_im(directory):
    image_dir = join(directory, "images")
    image_names = listdir(image_dir)
    image_path = join(image_dir, image_names[0])
    image = img.imread(image_path)

    return image[:, :, 0:3]


def outline_mask(mask, int_value=.5, boundary_value=1):
    output = mask * int_value
    num_rows, num_cols = mask.shape
    row, col = np.where(output == int_value)
    for i in range(len(row)):
        r = row[i]
        c = col[i]
        if r > 0:
            if mask[r-1, c] == 0:
                output[r, c] = boundary_value
                continue
        if r < num_rows-1:
            if mask[r+1, c] == 0:
                output[r, c] = boundary_value
                continue
        if c > 0:
            if mask[r, c-1] == 0:
                output[r, c] = boundary_value
                continue
        if c < num_cols-1:
            if mask[r, c+1] == 0:
                output[r, c] = boundary_value
                continue
    return output


def plot_im_and_mask(im, mask):
    clf()
    subplot(331)
    imshow(im)
    subplot(332)
    imshow(mask)
    subplot(333)
    imshow(rgb2gray(im))

    subplot(334)
    imshow(im[:, :, 0])
    subplot(335)
    imshow(im[:, :, 1])
    subplot(336)
    imshow(im[:, :, 2])

    nim = rgb_to_hsv(im)
    subplot(337)
    imshow(nim[:, :, 0])
    subplot(338)
    imshow(nim[:, :, 1])
    subplot(339)
    imshow(nim[:, :, 2])


def rgb2gray(rgb):
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return gray


def save_boundary_images():
    directory = training_path
    image_dir = listdir(directory)
    for image in image_dir:
        path = join(directory, image)
        im, mask_b = get_im_with_mask(path)
        boundary = np.copy(mask_b)
        boundary[boundary == 0.5] = 0
        mask = np.copy(mask_b)
        mask[mask == 0.5] = 1
        np.save(join(path, "boundary"), boundary)
        np.save(join(path, "mask_boundaries"), mask_b)
        np.save(join(path, "mask"), mask)


def generate_train_val_files(percent=0.2):
    directory = training_path
    train_path = 'training.txt'
    val_path = 'validation.txt'
    image_folders = listdir(directory)

    train_file = open(train_path, "w")
    val_file = open(val_path, "w")

    for image in image_folders:
        path = directory + "/" + image + "\n"
        if random.random() < percent:
            val_file.write(path)
        else:
            train_file.write(path)

    train_file.close()
    val_file.close()


def collapse_test():
    directory = training_path
    new_dir = 'all_test'
    if not exists(new_dir):
        makedirs(new_dir)
    name_list = listdir(directory)
    for name in name_list:
        src = join(join(directory, name), join('images', name + '.png'))
        dst = join(new_dir, name + '.png')
        copyfile(src, dst)


def word_to_hot_enc(word):
    labels = ["boundary", "background", "interior"]
    n = len(labels)
    hot = np.zeros(n)
    if word in labels:
        ind = labels.index(word)
        hot[ind] = 1
    else:
        hot = []
    return hot


def get_file_paths(directory):
    files = [directory + '/' + f for f in listdir(directory) if isfile(join(directory, f))]
    return files


def get_file_names(directory):
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    return files
