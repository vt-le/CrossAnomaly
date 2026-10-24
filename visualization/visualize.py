import os
import cv2
import glob
from PIL import Image
from visualization import html
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


class Visualize():
    def __init__(self, config, final_output_dir):
        self.web_dir = os.path.join(final_output_dir, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        mkdirs([self.web_dir, self.img_dir])
        self.win_size = 64
        self.display_id = 1
        self.name = config.DATA.DATASET

        use_tensorboard = False
        # tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter()

    def plot_current_errors(self, losses, step, name):
        # for tag, value in losses.items():
        #     self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
        self.writer.add_scalars(name,
                                {tag: value for tag, value in losses.items()},
                                step)

    def plot_current_images_to_board(self, image_list, step, name):
        self.writer.add_image('images', image_list, step)

    def display_current_results(self, visuals, epoch, step):
        for label, image_numpy in visuals.items():
            if image_numpy is None: continue
            ext = 'png' if 'label' in label else 'jpg'
            if isinstance(image_numpy, list):
                for i in range(len(image_numpy)):
                    img_path = os.path.join(self.img_dir, 'epoch%03d_iter%07d_%s_%d.%s' % (epoch, step, label, i, ext))
                    save_image(image_numpy[i], img_path)
            else:
                img_path = os.path.join(self.img_dir, 'epoch%03d_iter%07d_%s.%s' % (epoch, step, label, ext))
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                save_image(image_numpy, img_path)

        # update website
        webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=300)
        for n in range(epoch, 0, -1):
            webpage.add_header('epoch [%d]' % n)
            ims = []
            txts = []
            links = []

            for label, image_numpy in visuals.items():
                if image_numpy is None: continue
                ext = 'png' if 'label' in label else 'jpg'
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        if n == epoch:
                            img_path = 'epoch%03d_iter%07d_%s_%d.%s' % (n, step, label, i, ext)
                        else:
                            img_paths = sorted(
                                glob.glob(os.path.join(self.img_dir, 'epoch%03d_iter*_%s_%d.%s' % (n, label, i, ext))))
                            img_path = os.path.basename(img_paths[-1]) if len(img_paths) else 'img.jpg'
                        ims.append(img_path)
                        txts.append(label + str(i))
                        links.append(img_path)
                else:
                    if n == epoch:
                        img_path = 'epoch%03d_iter%07d_%s.%s' % (n, step, label, ext)
                    else:
                        img_paths = sorted(
                            glob.glob(os.path.join(self.img_dir, 'epoch%03d_iter*_%s.%s' % (n, label, ext))))
                        img_path = os.path.basename(img_paths[-1]) if len(img_paths) else 'img.jpg'
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
            if len(ims) < 6:
                webpage.add_images(ims, txts, links, width=self.win_size)
            else:
                num = int(round(len(ims) / 2.0))
                webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
        webpage.save()


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor[0, -1]
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor[:3]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def tensor2flow(output, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float().numpy()
    output = np.transpose(output, (1, 2, 0))
    #mag = np.max(np.sqrt(output[:,:,0]**2 + output[:,:,1]**2))
    #print(mag)
    hsv = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(output[..., 0], output[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb