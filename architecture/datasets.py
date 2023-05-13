import sys
sys.path.append('..')
import glob
import os
import numpy as np
import torch
from bvreader import binvox_rw
from torch.utils.data import Dataset, DataLoader

# Data Format:
# voxel array (true false occupancy), (p, sdf) uniform grid, (p, sdf) near surface, (p, sdf) on surface
class VoxelSDFDataset(Dataset):
    def __init__(self, split=1000):
        self.imgs_path = "../benches/"
        self.split = split
        
        file_list = glob.glob(self.imgs_path + "*")

        self.data = []
        
        first = False
        
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            self.data.append(class_name)
            
            if not first:
                npzfile = np.load('../benches/1a40eaf5919b1b3f3eaa2b95b99dae6/samples.npz')
                self.num_samples = npzfile['p_grid'].shape[0] // split
                first = True

        self.num_objs = len(self.data)
        
    def __len__(self):
        return self.split * len(self.data)
    
    
    def __getitem__(self, idx):
        which_split = idx // len(self.data)
        which_obj = idx % len(self.data)
        
        # load 100 points near surface
        # load 100 points uniformly sampled around :)
        obj = self.data[which_obj]
        
        voxel_path = os.path.join(self.imgs_path, obj, "model_watertight.binvox")
        sdf_path = os.path.join(self.imgs_path, obj, "samples.npz")
        
        obj_occ = None
        with open(voxel_path, 'rb') as f:
            obj_occ = binvox_rw.read_as_3d_array(f).data
        
        npzfile = np.load(sdf_path)
       
        p_grid = npzfile['p_grid'][which_split * self.num_samples:(which_split + 1) * self.num_samples]
        sdf_grid = npzfile['sdf_grid'][which_split * self.num_samples:(which_split + 1) * self.num_samples]
        data_grid = (p_grid, sdf_grid)
        
        p_near = npzfile['p_near'][which_split * self.num_samples:(which_split + 1) * self.num_samples]
        sdf_near = npzfile['sdf_near'][which_split * self.num_samples:(which_split + 1) * self.num_samples]
        data_near = (p_near, sdf_near)
        
        p_surf = npzfile['p_surf'][which_split * self.num_samples:(which_split + 1) * self.num_samples]
        sdf_surf = npzfile['sdf_surf'][which_split * self.num_samples:(which_split + 1) * self.num_samples]
        data_surf = (p_surf, sdf_surf)
        
        return [obj_occ, [data_grid, data_near, data_surf]]
    
    
    
def main():
    ds = VoxelSDFDataset()
    print(len(ds))
    
    counter = 0
    for item in ds:
        print(item)
    
if __name__ == '__main__':
    main()