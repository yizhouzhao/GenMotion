import torch
import os
from tqdm.auto import tqdm
import numpy as np

from .hdm05_params import ASF_JOINT2DOF

class HDM05Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, opt) -> None:
        super().__init__()
        self.opt = opt
        self.data_path = data_path
        self.frame_interval = opt.get("frame_interval", 10)
        self.input_motion_length = opt.get("input_motion_length",50)
        self.radius = opt.get("radius", True)

        self.amc_files = []
        self.animation_clips = []
        self.data = []

        self.load_amc()
        self.load_animation()
        self.load_data()
    
    def load_amc(self):
        for file in os.listdir(self.data_path):
            self.amc_files.append(file)
    
    def load_animation(self):
        print("loading amc files:")
        for amc_file in tqdm(self.amc_files):
            frame_count = 0 # record frame
            animation_clip = []
            with open(os.path.join(self.data_path, amc_file)) as f:
                new_frame_info = []
                for line in f.readlines():
                    if line.strip().isdigit():
                        frame_count += 1
                        
                        if len(new_frame_info) > 0:
                            animation_clip.append([_ for _ in new_frame_info])
                            new_frame_info.clear()

                    # set up animation,  interpolate every 20 key frames
                    elif frame_count > 0 and frame_count % self.frame_interval == 0:
                        
                        joint_name = line.strip().split()[0]
                        joint_pose = line.strip().split()[1:]

                        # TODO: change degree to radius
                        # if self.radius:  

                        new_frame_info.extend([float(_) for _ in joint_pose])
                
            self.animation_clips.append(animation_clip)
    
    def load_data(self):
        print("preparing training data")
        for clip in tqdm(self.animation_clips):
            if len(clip) < self.input_motion_length:
                continue

            for i in range(len(clip) - self.input_motion_length):
                self.data.append(clip[i: i + self.input_motion_length])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = torch.FloatTensor(self.data[idx])

        return item[:self.input_motion_length - 1], item[1:]
