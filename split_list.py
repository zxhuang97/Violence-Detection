import argparse
import os
import sys
import glob
import numpy as np
sys.path.append('.')

data_path = './data'
out_path = './'
video = list()
video.append(glob.glob(os.path.join(data_path,'Assault/*')))
video.append(glob.glob(os.path.join(data_path,'Fighting/*')))
video.append(glob.glob(os.path.join(data_path,'Robbery/*')))
video.append(glob.glob(os.path.join(data_path,'Shooting/*')))
train_video = glob.glob(os.path.join(data_path,'Training-Normal-Videos-Part-2/*'))
test_video =glob.glob(os.path.join(data_path,'Testing_Normal_Videos_Anomaly/*'))


for v in video:
	np.random.shuffle(v)
np.random.shuffle(test_video)

train_list = list()
test_list = list()

for v in video:
	total = len(v)
	test = total//10
	train_list.extend(v[test:])
	test_list.extend(v[:test])

train_list.extend(train_video)
test_list.extend(test_video)
print(len(train_list))
print(len(test_list))

train = open(os.path.join(out_path, 'train.txt'), 'w')
test = open(os.path.join(out_path, 'test.txt'), 'w')
for t in train_list:
	out = t.split('/')
	train.write(out[2] + '/' + out[3] + '\n')

for t in test_list:
	out = t.split('/')
	test.write(out[2] + '/' + out[3] + '\n')

train.close()
test.close()





