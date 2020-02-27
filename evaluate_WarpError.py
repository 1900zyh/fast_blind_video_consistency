#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import utils


def read_frame_from_videos(vname):
  frames = []
  vidcap = cv2.VideoCapture(vname)
  success, image = vidcap.read()
  count = 0
  while success:
    frames.append(image)
    success,image = vidcap.read()
    count += 1
  return frames

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')
    

    ### testing options
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')

    parser.add_argument('-dataset',         type=str,     required=True,            help='test datasets')
    parser.add_argument('-r', '--resume', default=None, type=str, required=True)
    parser.add_argument('-n', '--name', default=None, type=str, required=True)

    opts = parser.parse_args()
    print(opts)


    ## print average if result already exists
    metric_filename = os.path.join(opts.data_dir, opts.dataset, opts.name, "WarpError.txt")
    flow_warping = Resample2d().cuda()

    ### load video list
    print("===> Begin reading videos")
    if opts.name == 'orig':
        video_list = list(glob.glob(os.path.join(opts.resume, '*/orig.avi')))  
    else:
        video_list = list(glob.glob(os.path.join(opts.resume, '*/comp.avi')))

    print("===> Begin processing {} videos".format(len(video_list)))

    ### start evaluation
    err_all = np.zeros(len(video_list))

    for v, video in enumerate(video_list):
        video_name = video.split('/')[-2]

        occ_dir = os.path.join(opts.data_dir, opts.dataset, opts.name, "fw_occlusion", video_name)
        flow_dir =  os.path.join(opts.data_dir, opts.dataset, opts.name, "fw_flow", video_name)
        frame_list = read_frame_from_videos(video)

        err = 0
        for t in range(1, len(frame_list)):
            ### load input images
            img1 = frame_list[t-1]
            img2 = frame_list[t]
            print("Evaluate error for {}, {}/{} on frame {}/{}".format(video_name,
                v, len(video_list), t, len(frame_list)))

            ### load flow
            filename = os.path.join(flow_dir, "%05d.flo" %(t-1))
            flow = utils.read_flo(filename)

            ### load occlusion mask
            filename = os.path.join(occ_dir, "%05d.png" %(t-1))
            occ_mask = utils.read_img(filename)
            noc_mask = 1 - occ_mask

            with torch.no_grad():
                ## convert to tensor
                img2 = utils.img2tensor(img2).cuda()
                flow = utils.img2tensor(flow).cuda()
                ## warp img2
                warp_img2 = flow_warping(img2, flow)
                ## convert to numpy array
                warp_img2 = utils.tensor2img(warp_img2)

            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]
            err += np.sum(np.square(diff)) / N
        err_all[v] = err / (len(frame_list) - 1)
        print(err / (len(frame_list) - 1))
    print("\nAverage Warping Error = %f\n" %(err_all.mean()))

    err_all = np.append(err_all, err_all.mean())
    print("Save %s" %metric_filename)
    np.savetxt(metric_filename, err_all, fmt="%f")
