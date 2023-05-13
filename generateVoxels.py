import os
import numpy as np
import point_cloud_utils as pcu
import subprocess


subprocess.call(f"echo starting process...", shell=True) 
subprocess.call(f"pwd", shell=True) 
category_path = "./benches"

count = 0
for model_path in os.listdir(category_path):
    try:
        file = os.path.join(category_path, model_path, "model_watertight.obj")
        print(f'voxelizing {file}')
        # subprocess.call(f"cat {file}", shell=True) 
        subprocess.call(f"./binvox -d 2 {file}", shell=True) 
    except:
        print('error for file: {model_path}')
