import numpy as np
import cv2
import os
from PIL import Image

data_base_dir="/home/arakawa/BicycleGAN/datasets/hat_edge"
outfile_dir="/home/arakawa/BicycleGAN/datasets/hat"
processed_number=0
for file in os.listdir(data_base_dir):

    read_img_name=data_base_dir+'/'+file.strip()

    img=cv2.imread(read_img_name)

    while(1):
        imgg = np.zeros((256,256,3), np.uint8)
        
        
        processed_number+=1

        out_img_name=outfile_dir+'/'+file.strip()

        cv2.imwrite(out_img_name,255-imgg)

        break

