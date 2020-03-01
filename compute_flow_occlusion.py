#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

### custom lib
import networks
import utils


def read_frame_from_videos(vname):
  frames = []
  vidcap = cv2.VideoCapture(vname)
  success, image = vidcap.read()
  count = 0
  while success:
    image = image[:, :, ::-1] ## BGR to RGB
    image = np.float32(image) / 255.0
    frames.append(image)
    success,image = vidcap.read()
    count += 1
  return frames


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='optical flow estimation')

    ### testing options
    parser.add_argument('-model',           type=str,     default="FlowNet2",   help='Flow model name')
    parser.add_argument('-data_dir',        type=str,     default='data',       help='path to data folder')

    parser.add_argument('-dataset',         type=str,     required=True,        help='testing datasets')
    parser.add_argument('-r', '--resume', default=None, type=str, required=True)
    parser.add_argument('-n', '--name', default=None, type=str, required=True)


    opts = parser.parse_args()

    ### update options
    opts.grads = {} # dict to collect activation gradients (for training debug purpose)

    ### FlowNet options
    opts.rgb_max = 1.0
    opts.fp16 = False

    print(opts)

    ### initialize FlowNet
    print('===> Initializing model from %s...' %opts.model)
    model = networks.__dict__[opts.model](opts)

    ### load pre-trained FlowNet
    model_filename = os.path.join("pretrained_models", "%s_checkpoint.pth.tar" %opts.model)
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['state_dict'])
    print("===> Finished Loading model!")

    model = model.cuda()
    model.eval()

    ### load video list
    print("===> Begin reading videos")
    if opts.name == 'orig':
        video_list = list(glob.glob(os.path.join(opts.resume, '*/orig.avi')))  
    elif opts.name == 'post':
        video_list = list(glob.glob(os.path.join(opts.resume, '*/post.avi')))  
    else:
        video_list = list(glob.glob(os.path.join(opts.resume, '*/comp.avi')))

    print("===> Begin processing {} videos".format(len(video_list)))
    for vt, video in enumerate(video_list):
        video_name = video.split('/')[-2]
        fw_flow_dir = os.path.join(opts.data_dir, opts.dataset, opts.name, "fw_flow", video_name)
        if not os.path.isdir(fw_flow_dir):
            os.makedirs(fw_flow_dir)

        fw_occ_dir = os.path.join(opts.data_dir, opts.dataset, opts.name, "fw_occlusion", video_name)
        if not os.path.isdir(fw_occ_dir):
            os.makedirs(fw_occ_dir)

        fw_rgb_dir = os.path.join(opts.data_dir, opts.dataset, opts.name, "fw_flow_rgb", video_name)
        if not os.path.isdir(fw_rgb_dir):
            os.makedirs(fw_rgb_dir)

        frame_list = read_frame_from_videos(video)

        for t in range(len(frame_list) - 1):
            
            print("Compute flow for {}, {}/{} on frame {}/{}".format(video_name,
                vt, len(video_list), t, len(frame_list)))

            ### load input images 
            img1 = frame_list[t]
            img2 = frame_list[t+1]
            
            ### resize image
            size_multiplier = 64
            H_orig = img1.shape[0]
            W_orig = img1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)
            
            img1 = cv2.resize(img1, (W_sc, H_sc))
            img2 = cv2.resize(img2, (W_sc, H_sc))
        
            with torch.no_grad():
                ### convert to tensor
                img1 = utils.img2tensor(img1).cuda()
                img2 = utils.img2tensor(img2).cuda()
        
                ### compute fw flow
                fw_flow = model(img1, img2)
                fw_flow = utils.tensor2img(fw_flow)
            
                ### compute bw flow
                bw_flow = model(img2, img1)
                bw_flow = utils.tensor2img(bw_flow)


            ### resize flow
            fw_flow = utils.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig) 
            bw_flow = utils.resize_flow(bw_flow, W_out = W_orig, H_out = H_orig) 
            
            ### compute occlusion
            fw_occ = utils.detect_occlusion(bw_flow, fw_flow)

            ### save flow
            output_flow_filename = os.path.join(fw_flow_dir, "%05d.flo" %t)
            if not os.path.exists(output_flow_filename):
                utils.save_flo(fw_flow, output_flow_filename)
        
            ### save occlusion map
            output_occ_filename = os.path.join(fw_occ_dir, "%05d.png" %t)
            if not os.path.exists(output_occ_filename):
                utils.save_img(fw_occ, output_occ_filename)

            ### save rgb flow
            output_filename = os.path.join(fw_rgb_dir, "%05d.png" %t)
            if not os.path.exists(output_filename):
                flow_rgb = utils.flow_to_rgb(fw_flow)
                utils.save_img(flow_rgb, output_filename)

    print("===> Finished processing !")           




