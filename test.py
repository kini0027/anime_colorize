import numpy as np
import random
from random import uniform
import torch
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html

if __name__=='__main__':
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



    # test stage
    for i, data in enumerate(islice(dataset, opt.num_test)):
        model.set_input(data)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test))
        if not opt.sync:
            z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
            z_sample = z_samples.to('cpu').detach().numpy().copy()
            j = 0
            for j in range(31):
                while z_samples[j][1] > 1:
                    z_samples[j][1] = np.random.randn()
                    if z_samples[j][1] < 1:
                        break
            j = 0
            for j in range(31):
                while z_samples[j][2] == 0:
                    z_samples[j][2] = np.random.randn()
                    if z_samples[j][2] > 0 or z_samples[j][2] < 0:
                        break
        np.savetxt('05.txt', z_sample)


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
        print(z_samples)
        j = 0
        for j in range(31):
            print(z_samples[j][1])

    webpage.save()
