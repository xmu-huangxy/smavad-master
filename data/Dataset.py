import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import json


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


def np_load_frame(filename, resize_height, resize_width):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_height, resize_width)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0
    image_resized = np.transpose(image_resized, [2, 0, 1])
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, config, train=True, time_step=1, num_pred=1):
        self.train = train
        self.dir = config.PATH.TRAIN_FOLDER
        self.videos = OrderedDict()
        self._resize_height = config.DATA.IMG_SIZE
        self._resize_width = config.DATA.IMG_SIZE
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        if config.MODEL.TYPE == 'swin':
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):
                frames.append(self.videos[video_name]['frame'][i])

        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        for i in range(self._time_step + self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width)
            batch.append(image)
        if self.train:
            mask = self.mask_generator()
            return np.concatenate(batch, axis=0), mask
        else:
            return np.concatenate(batch, axis=0)

    def __len__(self):
        return len(self.samples)


class test_dataset:
    def __init__(self, config, video_folder):
        self.img_h = config.DATA.IMG_SIZE
        self.img_w = config.DATA.IMG_SIZE
        self.clip_length = 2
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs.sort()
        if config.MODEL.TYPE == 'swin':
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __len__(self):
        return len(self.imgs) - (self.clip_length - 1)  # The first [input_num] frames are unpredictable.

    def __getitem__(self, indice):
        video_clips = []
        for frame_id in range(indice, indice + self.clip_length):
            video_clips.append(np_load_frame(self.imgs[frame_id], self.img_h, self.img_w))
        video_clips = np.array(video_clips).reshape((-1, self.img_h, self.img_w))
        mask = self.mask_generator()
        return video_clips, mask


class GT_loader:
    def __init__(self, config):
        self.video_info = self.parser_videos_images_json(config.PATH.TEST_FOLDER, config.PATH.FRAME_MASK)
        self.cliplength = 2
        label = []
        for video_info in self.video_info.values():
            gt = self._load_json_gt_file(video_info)[(self.cliplength - 1)::]
            label.append(gt)
        self.label = np.concatenate(label, axis=0)

    def __test_gt__(self):
        return self.label

    def parser_videos_images_json(self, folder, frame_mask_file=''):  # folder: */frames
        print('parsing video json = {}'.format(frame_mask_file))
        videos_info = OrderedDict()
        with open(frame_mask_file, 'r') as file:
            data = json.load(file)

            for video_name in sorted(os.listdir(folder)):
                images_paths = glob.glob(os.path.join(folder, video_name, '*'))  # image*.jpg
                images_paths.sort()
                length = len(images_paths)

                assert length == data[video_name]['length']
                anomalies = data[video_name]['anomalies']

                frame_mask = []
                for event in anomalies:
                    for name, annotation in event.items():
                        frame_mask.append(annotation)

                videos_info[video_name] = {
                    'length': length,
                    'images': images_paths,
                    'frame_mask': frame_mask
                }
        print('parsing video successfully...')
        return videos_info

    @staticmethod
    def _load_json_gt_file(videoinfo):
        info = videoinfo
        anomalies = info['frame_mask']
        length = info['length']
        label = np.zeros((length,), dtype=np.int8)
        for event in anomalies:
            for annotation in event:
                start, end = annotation[0], annotation[1]
                label[start - 1: end] = 1
        gt = label
        return gt