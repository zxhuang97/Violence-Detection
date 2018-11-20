
import keras
import random
import numpy as np
import tensorflow as tf
from transform import trainAug
import cv2
import os
import time
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_file, modality = 'RGB',batch_size=1,seg=32, frames=16,dim=(112,112,3),image_pref='img_{:05d}.jpg',random_shift = True,transform=None, shuffle=True):
        'Initialization'
        self.list_file=list_file
        self.modality=modality

        self.batch_size = batch_size
        self.seg=seg
        self.frames=frames
        self.dim=dim
        self.image_pref=image_pref
        self.transform=transform

        self.random_shift =random_shift
        if self.random_shift:
            self.sample_indices = self.__get_train_indices
        else:
            self.sample_indices = self.__get_val_indices

        self.shuffle = shuffle
        self.video_list = self._parse_list()

        self.on_epoch_end()

    def _parse_list(self):
        video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        return video_list

    def __getitem__(self, index):
        'Generate one batch of data'
        t1 = time.time()
       
        videos = self.video_list[index*self.batch_size:(index+1)*self.batch_size]
        size = (self.batch_size,self.seg,self.frames,)+ self.dim

        x = np.zeros(size)
        y = np.zeros(self.batch_size)

        for i,video in enumerate(videos):
            offsets = self.sample_indices(video)
            x[i] = self.__data_generation(video,offsets)
            y[i] = video.label
        t2 = time.time()
        # print("Batch preparation time",t2-t1)

        return x,y

    def __get_train_indices(self, record):
        average_duration = (record.num_frames - self.frames +1) // self.seg
        offsets = np.multiply(list(range(self.seg)), average_duration) + np.random.randint(average_duration, size = self.seg)
        return offsets + 1

    def __get_val_indices(self,record):
        average_duration = (record.num_frames- self.frames +1) // self.seg
        offsets = np.multiply(list(range(self.seg)), average_duration) + average_duration //2
        return offsets + 1

    def __data_generation(self, record,indices):

        images=list()
        for seg_ind in indices:
            p = int(seg_ind)
            image=list()
            for i in range(self.frames):
                seg_imgs = self._load_image(record.path, p)
                image.append(seg_imgs)
                if p < record.num_frames:
                    p += 1
            image = self.transform(np.array(image))
            images.append(image)
        images = np.array(images)
        

        return images
    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return cv2.imread(os.path.join(directory, self.image_pref.format(idx)),cv2.IMREAD_COLOR)
        elif self.modality == 'Flow':
            x_img = cv2.imread(os.path.join(directory, self.image_pref.format('x', idx)),0)
            y_img = cv2.imread(os.path.join(directory, self.image_pref.format('y', idx)),0)

            return [x_img, y_img]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.video_list) / self.batch_size))

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.video_list)

