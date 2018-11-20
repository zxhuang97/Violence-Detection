import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import cv2

class trainAug(object):
	def __init__(self):
		self.cropScale= GroupMultiScaleCrop((320,240))
	def __call__(self,img_group):
		img_group=self.cropScale(img_group)
		img_group = groupFlip(img_group)
		img_group=groupNormolize(img_group)
		img_group=groupBright(img_group)
		return img_group

class valAug(object):

	def __call__(self,img_group):
		img_group = groupResize((112,112),img_group)
		img_group = groupNormolize(img_group)
		return img_group

def groupResize(size,img_group):
	out_images = list()
	for img in img_group:
		out_images.append(cv2.resize(img,dsize=size))
	return np.array(out_images)

def groupCrop(x0,y0,x1,y1,img_group):
	return img_group[:,y0:y1,x0:x1,:]

def groupNormolize(img_group):
	img_group = img_group / 255.0
	img_group -= [0.485, 0.456, 0.406]
	return img_group

def groupFlip(img_group):
	if random.random() <0.5 :
		return np.array([np.fliplr(x) for x in img_group])
	else:
		return img_group

def groupBright(img_group):
	img_group = img_group * np.random.uniform(1-0.15,1+0.15)
	return img_group


class GroupMultiScaleCrop(object):
	def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
		self.input_size=input_size
		self.scales=scales if scales is not None else [1, .875, .75, .66]
		self.max_distort=max_distort
		self.fix_crop=fix_crop
		self.more_fix_crop=more_fix_crop

		# base_size = min(input_size)
		# crop_w = [int(base_size * s) for s in self.scales]
		# crop_h = crop_w

		crop_w = [int(input_size[0] * s) for s in self.scales]
		crop_h = [int(input_size[1] * s) for s in self.scales]

		self.crop_pairs = []
		for i,h in enumerate(crop_h):
			for j,w in enumerate(crop_w):
				if abs(i-j) <= self.max_distort:
					self.crop_pairs.append((w,h))

	def __call__(self,img_group):

		crop_w, crop_h, offset_w, offset_h = self.__sample_crop_size()
		crop_img_group = groupCrop(offset_w, offset_h, crop_w+offset_w, crop_h+offset_h,img_group)
		crop_img_group = groupResize((112,112),crop_img_group)
		return crop_img_group

	def __sample_crop_size(self):
		crop_pair = random.choice(self.crop_pairs)

		if not self.fix_crop:
			w_offset = random.randint(0, self.input_size[0] - crop_pair[0])
			h_offset = random.randint(0, self.input_size[1] - crop_pair[1])
		else:
			w_offset, h_offset = self.sample_fix_offset(crop_pair)

		return crop_pair[0],crop_pair[1],w_offset,h_offset


	def sample_fix_offset(self,crop_pair):
		w_step = (self.input_size[0] - crop_pair[0]) // 4
		h_step = (self.input_size[1] - crop_pair[1]) // 4
		ret = list()
		ret.append((0, 0))  # upper left
		ret.append((4 * w_step, 0))  # upper right
		ret.append((0, 4 * h_step))  # lower left
		ret.append((4 * w_step, 4 * h_step))  # lower right
		ret.append((2 * w_step, 2 * h_step))  # center

		if self.more_fix_crop:
		    ret.append((0, 2 * h_step))  # center left
		    ret.append((4 * w_step, 2 * h_step))  # center right
		    ret.append((2 * w_step, 4 * h_step))  # lower center
		    ret.append((2 * w_step, 0 * h_step))  # upper center

		    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
		    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
		    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
		    ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
		return random.choice(ret)






