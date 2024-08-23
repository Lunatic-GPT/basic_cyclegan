import random
import time
import datetime
import sys
import os
from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class Logger:
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].data[0]

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


def saltpepper_noise(image, proportion):
    image_copy = image.copy()
    img_y, img_x = image.shape[0:2]
    x = np.random.randint(img_x, size=(int(proportion * img_x * img_y),))
    y = np.random.randint(img_y, size=(int(proportion * img_x * img_y),))
    image_copy[y, x] = np.random.choice([0, 255], size=(int(proportion * img_x * img_y), 1))

    sp_noise_plate = np.ones_like(image_copy) * 127
    sp_noise_plate[y, x] = image_copy[y, x]
    return image_copy, sp_noise_plate


class ImagesDataset_dicom_1channel_with_window_paird(Dataset):
    def __init__(self, folder, window, image_sz):
        super().__init__()
        self.folder = folder
        self.window = window
        self.image_sz = image_sz

        path_dcm = findAllFile(folder, ".dcm")
        paths = set()
        for path in path_dcm:
            paths.add(path)
        self.paths = list(paths)

        # print(f'{len(self.paths)} dicom images found')
        print("{} images found, using window size {}, image size {}".format(len(self.paths), self.window, image_sz))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        pos_paths = findAllFile(os.path.dirname(self.paths[index]), ".dcm")
        neg_paths = findAllFile(os.path.dirname(self.paths[index]).replace("pos", "neg"), ".dcm")

        dcm_pos1 = sitk.ReadImage(self.paths[index])
        img_pos1 = sitk.GetArrayFromImage(dcm_pos1)

        if len(neg_paths) == 0:
            img_flatten = img_pos1.flatten().copy()
            lst = range(0, 144 * 144)  # 因为包含0和144，所以是145

            # 随机选择1000个不重复的数字
            random_numbers = random.sample(lst, 2000)

            for i in range(len(random_numbers)):
                img_flatten[random_numbers[i]] = np.mean(img_flatten)

            img_neg = np.reshape(img_flatten, [1, 144, 144])
            # plt.imshow(np.squeeze(img_neg))
            # plt.show()
        else:
            dcm_neg = sitk.ReadImage(random.choice(neg_paths))
            img_neg = sitk.GetArrayFromImage(dcm_neg)

        img_pos1[img_pos1 > self.window] = self.window
        img_neg[img_neg > self.window] = self.window

        img_pos1 = min_max_norm(img_pos1)
        img_neg = min_max_norm(img_neg)
        return torch.from_numpy(img_pos1.astype(np.float32)), torch.from_numpy(img_neg.astype(np.float32))


class ImagesDataset_dicom_1channel_with_window_cycleGAN_unpaird(Dataset):
    def __init__(self, folder, window, image_sz):
        super().__init__()
        self.folder = folder
        self.window = window
        self.image_sz = image_sz

        path_dcm = findAllFile(folder, ".dcm")
        paths = set()
        for path in path_dcm:
            paths.add(path)
        self.paths = list(paths)

        # print(f'{len(self.paths)} dicom images found')
        print("{} images found, using window size {}, image size {}".format(len(self.paths), self.window, image_sz))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        pos_folder = self.folder
        neg_paths = findAllFile(pos_folder.replace("pos", "neg"), ".dcm")

        dcm_pos1 = sitk.ReadImage(self.paths[index])
        img_pos1 = sitk.GetArrayFromImage(dcm_pos1)

        dcm_neg = sitk.ReadImage(random.choice(neg_paths))
        img_neg = sitk.GetArrayFromImage(dcm_neg)

        img_pos1[img_pos1 > self.window] = self.window
        img_neg[img_neg > self.window] = self.window

        img_pos1 = min_max_norm(img_pos1)
        img_neg = min_max_norm(img_neg)
        return torch.from_numpy(img_pos1.astype(np.float32)), torch.from_numpy(img_neg.astype(np.float32))


def min_max_norm(src_arr):
    max_val = np.max(np.max(src_arr))
    min_val = np.min(np.min(src_arr))

    norm_arr = (src_arr - min_val) / (max_val - min_val + 1e-10)

    return norm_arr


def center_crop_or_pad_array(original_array, sz=256):
    # 获取原始数组的尺寸
    original_rows = len(original_array)
    original_cols = len(original_array[0])

    # 确定新数组的尺寸，如果原始数组小于256，则填充到256x256
    # 如果原始数组大于256，则裁剪到256x256
    new_rows = min(sz, original_rows)
    new_cols = min(sz, original_cols)

    # 计算裁剪的起始点
    start_row = (original_rows - new_rows) // 2 if original_rows > new_rows else 0
    start_col = (original_cols - new_cols) // 2 if original_cols > new_cols else 0

    # 创建一个新的数组，其尺寸为256x256，初始填充为0
    new_array = [[0 for _ in range(sz)] for _ in range(sz)]

    # 根据需要裁剪或填充
    for i in range(new_rows):
        for j in range(new_cols):
            # 将原始数组的中心部分复制到新数组
            new_array[i][j] = original_array[start_row + i][start_col + j]

    return np.array(new_array)


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def findAllFile(path, end_name):
    assert end_name[0] == '.' or end_name[0] == "."
    filelist = []
    for root, ds, fs in os.walk(path):
        for f in fs:
            if f.endswith(end_name):
                fullname = os.path.join(root, f)
                filelist.append(fullname)
    return filelist


class ImagesDataset_dicom_1channel_with_window_paird_infer(Dataset):
    def __init__(self, folder, window, image_sz):
        super().__init__()
        self.folder = folder
        self.window = window
        self.image_sz = image_sz

        path_dcm = findAllFile(folder, ".dcm")
        paths = set()
        for path in path_dcm:
            paths.add(path)
        self.paths = list(paths)

        # print(f'{len(self.paths)} dicom images found')
        print("{} images found, using window size {}, image size {}".format(len(self.paths), self.window, image_sz))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        dcm_pos1 = sitk.ReadImage(self.paths[index])

        img_pos1 = sitk.GetArrayFromImage(dcm_pos1)
        img_pos1[img_pos1 > self.window] = self.window
        img_pos1 = min_max_norm(img_pos1)

        return torch.from_numpy(img_pos1.astype(np.float32)), self.paths[index]
