import numpy as np
import cv2
import os
from PIL import Image

data_base_dir="/home/arakawa/BicycleGAN/datasets/edges2shoes/train"
outfile_dir="/home/arakawa/BicycleGAN/cropped"
processed_number=0
for file in os.listdir(data_base_dir):

    read_img_name=data_base_dir+'/'+file.strip()

    img=cv2.imread(read_img_name)

    while(1):
        cropped = img[0:256,256:512]        
        
        processed_number+=1

        out_img_name=outfile_dir+'/'+file.strip()

        cv2.imwrite(out_img_name,cropped)

        break

