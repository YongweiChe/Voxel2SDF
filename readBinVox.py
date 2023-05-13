import numpy as np
from bvreader import binvox_rw


if __name__ == '__main__':
    print("hi")
    
    with open('bvreader/chair.binvox', 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        print(model.dims)
        print(model.scale)
        print(model.data[0])
        print(np.sum(model.data))
    
