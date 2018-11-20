import argparse
import os
import sys

sys.path.append('.')
from utils import parse_directory, build_split_list,parse_split_file

parser = argparse.ArgumentParser()
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y')
parser.add_argument('--out_list_path', type=str, default='data/')
parser.add_argument('--shuffle', action='store_true', default=False)

args = parser.parse_args()

frame_path = args.frame_path
rgb_p = args.rgb_prefix
flow_x_p = args.flow_x_prefix
flow_y_p = args.flow_y_prefix
out_path = args.out_list_path
shuffle = args.shuffle
trainlist_path = '/home/marcus/violence-detection/data/train.txt'
testlist_path = '/home/marcus/violence-detection/data/test.txt'
print(os.getcwd())
# operation
print ('processing dataset')
#return the (vid,label) tuple of each split
split_tp = parse_split_file(trainlist_path,testlist_path)
print(len(split_tp[0]),len(split_tp[1]))
#return (name_to_path,name_to_rgbcounts,name_to_flowcounts)
f_info = parse_directory(frame_path, rgb_p, flow_x_p, flow_y_p)
print(len(f_info[0]))
print(split_tp[0][0][0])
print(f_info[0][split_tp[0][0][0]])

print ('writing list files for training/testing')
# lists = build_split_list(split_tp, f_info, shuffle)
# open(os.path.join(out_path, 'rgb_train.txt'), 'w').writelines(lists[0][0])
# open(os.path.join(out_path, 'rgb_val.txt'), 'w').writelines(lists[0][1])
# open(os.path.join(out_path, 'flow_train.txt'), 'w').writelines(lists[1][0])
# open(os.path.join(out_path, 'flow_val.txt'), 'w').writelines(lists[1][1])

