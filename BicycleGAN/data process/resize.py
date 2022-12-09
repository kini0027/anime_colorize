import cv2
import os
from PIL import Image  
import os.path  
import glob  

data_base_dir="/home/arakawa/BicycleGAN/datasets/hat"
outfile_dir="/home/arakawa/BicycleGAN/datasets/hat"
processed_number=0
for file in os.listdir(data_base_dir):

    read_img_name=data_base_dir+'/'+file.strip()

    image=cv2.imread(read_img_name)
    

    while(1):
        
        imggg=cv2.resize(image,(256,256))
        processed_number+=1

        out_img_name=outfile_dir+'/'+file.strip()

        cv2.imwrite(out_img_name,imggg)

        break

