# Some code adapted from
# https://github.com/brendenlake/omniglot/blob/master/python/demo.py

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


root_dir = ('/Users/tuananhle/Documents/research/datasets/'
            'brendenlake/omniglot/python/')

train_img_dir = os.path.join(root_dir, 'images_background')
train_motor_dir = os.path.join(root_dir, 'strokes_background')

train_img_small_1_dir = os.path.join(root_dir, 'images_background_small1')
train_img_small_2_dir = os.path.join(root_dir, 'images_background_small2')
train_motor_small_1_dir = os.path.join(root_dir, 'strokes_background_small1')
train_motor_small_2_dir = os.path.join(root_dir, 'strokes_background_small2')

test_img_dir = os.path.join(root_dir, 'images_evaluation')
test_motor_dir = os.path.join(root_dir, 'strokes_evaluation')


def filter_listdir(listdir):
    """Filters items in dir which don't start with a dot"""
    return [x for x in listdir if x[0] != '.']


def num2str(idx):
    """Convert to str and add leading zero to single
    digit numbers"""
    if idx < 10:
        return '0' + str(idx)
    return str(idx)


def plot_img(ax, img):
    """Plot image.

    Args:
        ax
        img: np.array [105 x 105] grayscale image"""
    ax.imshow(img, cmap='gray')
    return ax


def plot_motor(ax, motor, lw=2):
    """Plot the motor trajectory

    Args:
        ax
        motor: [ns list] of strokes (numpy arrays) in motor space
        lw : line width
    """
    # strip off the timing data (third column)
    motor_ = [d[:, 0:2] for d in motor]
    # convert to image space
    motor_ = [space_motor_to_img(d) for d in motor]
    num_strokes = len(motor_)
    # for each stroke
    for stroke_id in range(num_strokes):
        plot_traj(ax, motor_[stroke_id], get_color(stroke_id), lw)
    return ax


def plot_traj(ax, stroke, color, lw):
    """Plot individual stroke

    Args:
        ax
        stroke: [n x 2] individual stroke
        color: stroke color
        lw: line width"""
    num_points = stroke.shape[0]
    if num_points > 1:
        ax.plot(stroke[:, 0], stroke[:, 1], color=color, linewidth=lw)
    else:
        ax.plot(stroke[0, 0], stroke[0, 1], color=color, linewidth=lw,
                marker='.')


def space_motor_to_img(pt):
    """Map from motor space to image space (or vice versa)

    Args:
        pt: [n x 2] points (rows) in motor coordinates

    Output:
        new_pt: [n x 2] points (rows) in image coordinates
    """
    new_pt = pt.copy()
    new_pt[:, 1] = -pt[:, 1]
    return new_pt


def get_color(k):
    """Color map for the stroke of index k"""

    scol = ['r', 'g', 'b', 'm', 'c']
    ncol = len(scol)
    if k < ncol:
        out = scol[k]
    else:
        out = scol[-1]
    return out


def load_img(path):
    """Load binary image for a character

    Args:
        path"""
    img = plt.imread(path)
    img = np.array(img, dtype=bool)
    return img


def load_motor(path):
    """Load motor data for a character from text file

    Args:
      path

    Returns:
      motor: list of strokes (each is a [n x 3] numpy array)
         first two columns are coordinates
         the last column is the timing data (in milliseconds)
    """

    motor = []
    with open(path, 'r') as fid:
        lines = fid.readlines()
    lines = [l.strip() for l in lines]
    for myline in lines:
        if myline == 'START':  # beginning of character
            stroke = []
        elif myline == 'BREAK':  # break between strokes
            stroke = np.array(stroke)
            motor.append(stroke)  # add to list of strokes
            stroke = []
        else:
            arr = np.fromstring(myline, dtype=float, sep=',')
            stroke.append(arr)
    return motor


def get_example(img_dir, motor_dir, alphabet_id, char_id, example_id):
    alphabets = sorted(filter_listdir(os.listdir(img_dir)))
    alphabet = alphabets[alphabet_id]

    img_char_dir = os.path.join(img_dir, alphabet, 'character' +
                                num2str(char_id + 1))
    motor_char_dir = os.path.join(motor_dir, alphabet, 'character' +
                                  num2str(char_id + 1))

    example_filename = filter_listdir(os.listdir(img_char_dir))[0]
    example_base = example_filename[:example_filename.find('_')]

    example_img_path = os.path.join(img_char_dir, example_base + '_' +
                                    num2str(example_id + 1) + '.png')
    example_motor_path = os.path.join(motor_char_dir, example_base + '_' +
                                      num2str(example_id + 1) + '.txt')
    example_img = load_img(example_img_path)
    example_motor = load_motor(example_motor_path)

    return example_img, example_motor


def get_num_alphabets(img_dir):
    alphabets = sorted(filter_listdir(os.listdir(img_dir)))
    return len(alphabets)


def get_num_chars(img_dir, alphabet_id):
    alphabets = sorted(filter_listdir(os.listdir(img_dir)))
    alphabet = alphabets[alphabet_id]
    alphabet_dir = os.path.join(img_dir, alphabet)

    char_dirs = sorted(filter_listdir(os.listdir(alphabet_dir)))
    return len(char_dirs)


def get_num_examples(img_dir, alphabet_id, char_id):
    alphabets = sorted(filter_listdir(os.listdir(img_dir)))
    alphabet = alphabets[alphabet_id]
    img_char_dir = os.path.join(img_dir, alphabet, 'character' +
                                num2str(char_id + 1))
    example_filenames = filter_listdir(os.listdir(img_char_dir))
    return len(example_filenames)


def load_imgs_and_motors(img_dir, motor_dir):
    imgs = []
    motors = []

    for alphabet_id in tqdm(range(get_num_alphabets(img_dir))):
        imgs.append([])
        motors.append([])
        for char_id in range(get_num_chars(img_dir, alphabet_id)):
            imgs[alphabet_id].append([])
            motors[alphabet_id].append([])
            for example_id in range(get_num_examples(img_dir, alphabet_id,
                                    char_id)):
                example_img, example_motors = get_example(
                    img_dir, motor_dir, alphabet_id, char_id, example_id)
                imgs[alphabet_id][char_id].append(example_img)
                motors[alphabet_id][char_id].append(example_motors)

    return imgs, motors


class OmniglotDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, motor_dir, device=torch.device('cpu'),
                 only_img=True):
        imgs_loaded, motors_loaded = load_imgs_and_motors(img_dir, motor_dir)
        self.only_img = only_img
        self.imgs = []
        if not self.only_img:
            self.motors = []

        for alphabet_id in range(len(imgs_loaded)):
            for char_id in range(len(imgs_loaded[alphabet_id])):
                # each character has exactly 20 example imgs
                self.imgs.append(torch.cat([
                    torch.tensor(1 - img, dtype=torch.float,
                                 device=device).unsqueeze(0)
                    for img in imgs_loaded[alphabet_id][char_id]]))
                if not self.only_img:
                    self.motors.append(motors_loaded[alphabet_id][char_id])

        del imgs_loaded
        del motors_loaded

    def __getitem__(self, idx):
        if self.only_img:
            return self.imgs[idx]
        else:
            return self.imgs[idx], self.motors[idx]

    def __len__(self):
        return len(self.imgs)


def omniglot_dataset_collate_fn(list_of_imgs_and_motors):
    imgss = []
    motorss = []
    for imgs, motors in list_of_imgs_and_motors:
        imgss.append(imgs.unsqueeze(0))
        motorss.append(motorss)
    return torch.cat(imgss), motorss


def get_omniglot_dataloader(img_dir, motor_dir, batch_size, shuffle,
                            only_img=True):
    omniglot_dataset = OmniglotDataset(img_dir, motor_dir, only_img=only_img)
    if only_img:
        collate_fn = None
    else:
        collate_fn = omniglot_dataset_collate_fn
    return torch.utils.data.DataLoader(
        omniglot_dataset, batch_size, shuffle=shuffle,
        collate_fn=collate_fn)
