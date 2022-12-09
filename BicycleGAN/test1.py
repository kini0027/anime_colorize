import numpy as np
import random
import torch
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html


# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    
    #np.savetxt(''home/arakawa/BicycleG/01.txt'',z_samples)
    #np.savetxt('01.txt',z_samples,fmt='%0.8f')

# test stage
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
        z_sample = z_samples.to('cpu').detach().numpy().copy()
        np.savetxt('04.txt',z_sample)
        #z_np = np.array([[-2.440417408943176270e-01,-1.170862793922424316e+00,-6.431127786636352539e-01,-1.170862793922424316e+00,4.199393838644027710e-02,1.209136962890625000e+00,6.233549118041992188e-01,1.136510848999023438e+00],[-9.403626918792724609e-01,1.170862793922424316e+00,-6.431127786636352539e-01,-2.122577905654907227e+00,3.688648939132690430e-01,6.877452135086059570e-01,-4.773858785629272461e-01,-4.309330284595489502e-01]])
        z_np = np.loadtxt('05.txt')
        z_sample = np.loadtxt('04.txt')
        arr = z_sample
        arr[1] = z_np[4]
        arr[2] = z_np[13]
        arr[3] = z_np[30]
        #z_np[4] = z_np[9]
        
        '''
        #一点交叉 6
        j = 0
        arr[4] = arr[1]
        arr[5] = arr[2]
        a = random.randint(4,8)
        for j in range(a,8):
            b = arr[4][j]
            arr[4][j] = arr[5][j]
            arr[5][j] = b
        j = 0
        for j in range(a):
            #c = np.random.uniform(0,1)
            #if c < 0.5:
            arr[4][j] = z_sample[4][j]
            arr[5][j] = z_sample[5][j]
        j = 0
        arr[6] = arr[1]
        arr[7] = arr[3]
        a = random.randint(4,8)
        for j in range(a,8):
            b = arr[6][j]
            arr[6][j] = arr[7][j]
            arr[7][j] = b
        j = 0
        for j in range(a):
            #c = np.random.uniform(0,1)
            #if c < 0.5:
            arr[6][j] = z_sample[6][j]
            arr[7][j] = z_sample[7][j]
        j = 0
        arr[8] = arr[2]
        arr[9] = arr[3]
        a = random.randint(4,8)
        for j in range(a,8):
            b = arr[8][j]
            arr[8][j] = arr[9][j]
            arr[9][j] = b
        j = 0
        for j in range(a):
            #c = np.random.uniform(0,1)
            #if c < 0.5:
            arr[8][j] = z_sample[8][j]
            arr[9][j] = z_sample[9][j]
        j = 0
        '''
        #一样交叉 6
        arr[4] = arr[1]
        arr[5] = arr[2]
        for j in range(8):
            c = np.random.uniform(0,1)
            if c < 0.5:
                b = arr[4][j]
                arr[4][j] = arr[5][j]
                arr[5][j] = b
            else:
                arr[4][j] = z_sample[4][j]
                arr[5][j] = z_sample[5][j]
        j = 0
        arr[6] = arr[1]
        arr[7] = arr[3]
        for j in range(8):
            c = np.random.uniform(0,1)
            if c < 0.5:
                b = arr[6][j]
                arr[6][j] = arr[7][j]
                arr[7][j] = b
            else:
                arr[6][j] = z_sample[6][j]
                arr[7][j] = z_sample[7][j]
        j = 0
        arr[8] = arr[2]
        arr[9] = arr[3]
        for j in range(8):
            c = np.random.uniform(0,1)
            if c < 0.5:
                b = arr[8][j]
                arr[8][j] = arr[9][j]
                arr[9][j] = b
            else:
                arr[8][j] = z_sample[8][j]
                arr[9][j] = z_sample[9][j]
        j = 0
        #平均 3
        for j in range(8):
            arr[10][j] = (arr[1][j]+arr[2][j])/2
        j = 0
        for j in range(8):
            arr[11][j] = (arr[2][j]+arr[3][j])/2
        j = 0        
        for j in range(8):
            arr[12][j] = (arr[1][j]+arr[3][j])/2
        j = 0   

#        for j in range(8):
#           arr[13][j] = (arr[1][j]+arr[2][j]+arr[3][j])/3
        #外分 3
        a = 0.7
        for j in range(8):
            arr[13][j] = arr[3][j]+a*(arr[3][j]-arr[1][j])
            while arr[13][j] < -1 or arr[13][j] > 1:
                #arr[13][j] = z_sample[13][j]
                arr[13][j] = np.random.randn()

        j = 0
        for j in range(8):   
            arr[14][j] = arr[1][j]+a*(arr[1][j]-arr[2][j])
            while arr[14][j] < -1 or arr[14][j] > 1:
            	#arr[14][j] = z_sample[14][j]
                #arr[14][j] = random.random()
                arr[14][j] = np.random.randn()
        j = 0
        for j in range(8):   
            arr[15][j] = arr[2][j]+a*(arr[2][j]-arr[3][j])
            while arr[15][j] < -1 or arr[15][j] > 1:
                #arr[15][j] = z_sample[15][j]
                arr[15][j] = np.random.randn()
        j = 0

#        for j in range(8):   
#            arr[21][j] = z_np[1][j]+a*(z_np[1][j]-z_np[4][j])
#        j = 0
#        for j in range(8):   
#            arr[22][j] = z_np[2][j]+a*(z_np[2][j]-z_np[4][j])
#        j = 0
        #一样交叉 6
        arr[16] = arr[13]
        arr[17] = arr[14]
        for j in range(8):
            c = np.random.uniform(0,1)
            if c < 0.5:
                b = arr[16][j]
                arr[16][j] = arr[17][j]
                arr[17][j] = b
            else:
                arr[16][j] = z_sample[16][j]
                arr[17][j] = z_sample[17][j]
        j = 0
        arr[18] = arr[13]
        arr[19] = arr[15]
        for j in range(8):
            c = np.random.uniform(0,1)
            if c < 0.5:
                b = arr[18][j]
                arr[18][j] = arr[19][j]
                arr[19][j] = b
            else:
                arr[18][j] = z_sample[18][j]
                arr[19][j] = z_sample[19][j]
        j = 0
        arr[20] = arr[14]
        arr[21] = arr[15]
        for j in range(8):
            c = np.random.uniform(0,1)
            if c < 0.5:
                b = arr[20][j]
                arr[20][j] = arr[21][j]
                arr[21][j] = b
            else:
                arr[20][j] = z_sample[20][j]
                arr[21][j] = z_sample[21][j]
        j = 0
        '''
        #新たな交叉 3
        arr[16] = arr[1]
        arr[17] = arr[2]
        arr[18] = arr[3]
        arr[19] = arr[10]
        arr[20] = arr[11]
        arr[21] = arr[12]
        arr[22] = arr[13]
        for j in range(8):
            c = np.random.uniform(0,1)
            if c > 0.5:
                arr[16][j] = z_sample[16][j]
        j = 0
        for j in range(8):
            c = np.random.uniform(0,1)
            if c > 0.5:
                arr[17][j] = z_sample[17][j]
        j = 0
        for j in range(8):
            c = np.random.uniform(0,1)
            if c > 0.5:
                arr[18][j] = z_sample[18][j]
        j = 0
        for j in range(8):
            c = np.random.uniform(0,1)
            if c > 0.5:
                arr[19][j] = z_sample[19][j]
        j = 0
        for j in range(8):
            c = np.random.uniform(0,1)
            if c > 0.5:
                arr[20][j] = z_sample[20][j]
        j = 0
        for j in range(8):
            c = np.random.uniform(0,1)
            if c >0.5:
                arr[21][j] = z_sample[21][j]
        j = 0
        for j in range(8):
            c = np.random.uniform(0,1)
            if c > 0.5:
                arr[22][j] = z_sample[22][j]
        j = 0
        '''
        for j in range(22,31):
            arr[j] = z_sample[j]
        z_sample = arr
        np.savetxt('05.txt',z_sample)
        z_sample = z_sample.astype(np.float32)
        z_tensor = torch.from_numpy(z_sample).clone()
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        z_samples = z_tensor.to('cuda')
        
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)

    img_path = 'input_%3.3d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

webpage.save()
