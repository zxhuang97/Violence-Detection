import glob
import fnmatch
import os
import random
import tqdm

def build_split_list(split_tuple, frame_info, shuffle=False):
    split = split_tuple

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            frame_dir = frame_info[0][item[0]]
            rgb_cnt = frame_info[1][item[0]]
            flow_cnt = frame_info[2][item[0]]
            rgb_list.append('{} {} {}\n'.format(frame_dir, rgb_cnt, item[1]))
            flow_list.append('{} {} {}\n'.format(frame_dir, flow_cnt, item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list

    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)

def parse_split_file(train_path,test_path):

    def line2rec(line):
        items = line.strip().split('/')
        label = 1
        if fnmatch.fnmatch(items[0],"*Normal*"):
            label = 0
        vid = items[1].split('.')[0]
        return vid, label

    train_list = [line2rec(x) for x in open(train_path)]
    test_list = [line2rec(x) for x in open(test_path)]

    return (train_list,test_list)

def parse_directory(path, rgb_prefix='img_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames 
    Returns:
      a tuple holding 3 dict (name_to_path,name_to_rgbcounts,name_to_flowcounts)
    """
    print ('parse frames under folder {}'.format(path))
    print(path)
    frame = path + '*'
    print(frame)
    frame_folders = glob.glob(frame)
    #frame_folders = glob.glob('/media/marcus/violence-detection/frame/*')
    print("there are {} folders".format(len(frame_folders)))

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    # check RGB
    rgb_counts = {}
    flow_counts = {}
    dir_dict = {}
    for i,f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = f.split('/')[-1]
        rgb_counts[k] = all_cnt[0]
        dir_dict[k] = f
        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: '+f)
        flow_counts[k] = x_cnt

    print ('frame folder analysis done')
    return dir_dict, rgb_counts, flow_counts
