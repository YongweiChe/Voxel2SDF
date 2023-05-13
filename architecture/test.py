import sys
sys.path.append('..')

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import numpy as np
import sys
from bvreader import binvox_rw

from models import Encoder, Decoder

from datasets import VoxelSDFDataset

def implicit_f(point, latent, decoder): # point is a torch.tensor([[x,y,z]])
    return float(decoder(torch.cat((latent, point), dim=1)))


def create_binvox(latent, decoder, dim=16):
    # double for loop it up in here
    x_arr =np.linspace(-0.5, 0.5,num=dim,endpoint=True)
    y_arr =np.linspace(-0.5, 0.5,num=dim,endpoint=True)
    z_arr =np.linspace(-0.5, 0.5,num=dim,endpoint=True)
    
    occupancy_arr = np.zeros((dim, dim, dim))
    
    for idx, x in enumerate(x_arr):
        for idy, y in enumerate(y_arr):
            for idz, z in enumerate(z_arr):
                print(x,y,z)
                point = torch.tensor([[float(x), float(y), float(z)]])
                if implicit_f(point, latent, decoder) < 0:
                    occupancy_arr[idx, idy, idz] = 1
    
    # load dummy binvox
    with open('../bvreader/chair.binvox', 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        model.data = occupancy_arr
        with open('./genvox/chair.binvox', 'wb') as new_f:
            model.write(new_f)

def main():
    encoder = torch.load('./save/encoder_l16_c3_e0.pt')
    decoder = torch.load('./save/decoder_l16_c3_e0.pt')
    
    latent = encoder(torch.randn(1,1,64,64,64))
    print(latent)
    
    point = torch.tensor([[0.25, 0.25, 0.25]])
    
    print(implicit_f(point, latent, decoder))
    
    create_binvox(latent, decoder)
    
    

if __name__ == '__main__':
    main()