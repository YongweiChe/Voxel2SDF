import numpy as np


# (p_vol, sdf_vol) correspond to point and sdf
# (p_surf, n_surf) correspond to a point on the surface and its surface normal
def main(): 
    npzfile = np.load('./benches/1a40eaf5919b1b3f3eaa2b95b99dae6/samples.npz')
    print(npzfile.files)
    print('-----p_grid-----')
    print(npzfile['p_grid'])
    print(np.shape(npzfile['p_grid']))
    
    print('-----sdf_grid-----')
    print(npzfile['sdf_vol'])
    print(np.shape(npzfile['sdf_grid']))
    
    print('-----p_near-----')
    print(npzfile['p_near'])
    print(np.shape(npzfile['p_near']))
    
    print('-----sdf_near-----')
    print(npzfile['sdf_near'])
    print(np.shape(npzfile['sdf_near']))
    
    print('-----p_surf-----')
    print(npzfile['p_surf'])
    print(np.shape(npzfile['p_surf']))
    
    print('-----sdf_surf-----')
    print(npzfile['sdf_surf'])
    print(np.shape(npzfile['sdf_surf']))


if __name__ == '__main__':
    main()