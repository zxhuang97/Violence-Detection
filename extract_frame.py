__author__ = 'yjxiong'

import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process
import cv2
import argparse
out_path = ''

def dump_frames(vid_path):
    import cv2
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    for i in range(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        
    print ('{} done'.format(vid_name))
    

def run_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_gpu')+' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
        quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])

    os.system(cmd)
    print ('{} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--num_worker", type=int, default=20)
    parser.add_argument("--df_path", type=str, default='/home/marcus/temporal-segment-networks/lib/dense_flow/', help='path to the dense_flow toolbox')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='path to the dense_flow toolbox')
    parser.add_argument("--ext", type=str, default='mp4', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=320, help='resize image width')
    parser.add_argument("--new_height", type=int, default=240, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=1, help='number of GPU')

    args = parser.parse_args()
    print("PID of main process is ",os.getpid())
    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    df_path = args.df_path
    out_format = args.out_format
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu
    print(num_worker)
    if not os.path.isdir(out_path):
        print ("creating folder: " + out_path)
        os.makedirs(out_path)

    # vid_list = glob.glob(src_path + '/*/*.'+ext)
    # print (len(vid_list))
    pool = Pool(num_worker)
    # for a in vid_list:
    #     dump_frames(a)
    vid_list =list()
    vid_list.append(src_path + '/Training-Normal-Videos-Part-2/Normal_Videos547_x264.mp4')
    vid_list.append(src_path + '/Training-Normal-Videos-Part-2/Normal_Videos633_x264.mp4')
    vid_list.append(src_path + '/Training-Normal-Videos-Part-2/Normal_Videos946_x264.mp4')
    pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))
    #pool.map(dump_frames,vid_list)
