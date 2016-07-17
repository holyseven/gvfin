import caffe

import numpy as np
from PIL import Image

import random


class C3DImageNetInputLayer(caffe.Layer):
    """
    """

    def setup(self, bottom, top):
        """
        parameters:

        - imagelist
        - labellist
        - mean
        - randomize (True)
        - seed (None)
        - new_H
        - new_W
        - batch_size
        
        """
        # config
        params = eval(self.param_str)
        self.imagelist = params['imagelist']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.new_H = params.get('new_H', 112) # 112+32
        self.new_W = params.get('new_W', 112) # 112+32
        self.batch_size = params.get('batch_size', 1)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        # load address of images and labels
        data_lines = open(self.imagelist, 'r').read().splitlines()
        
        self.imageindices = [None] * len(data_lines)
        self.labelindices = [None] * len(data_lines)

        for i in range(len(data_lines)):
            line = data_lines[i].split(' ')
            self.imageindices[i] = line[0]
            self.labelindices[i] = line[1]

        self.idx = 0
        # random
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.labelindices)-1)

    def reshape(self, bottom, top):
        n = self.batch_size
        self.data, self.label = self.load_video_and_label(self.idx)
        if n == 1:
            top[0].reshape(*self.data.shape)
            top[1].reshape(*self.label.shape)
        else:
            n = n - 1
            while n > 0:
                self.idx = random.randint(0, len(self.labelindices)-1)
                _data, _label = self.load_video_and_label(self.idx)
                self.data = np.concatenate((self.data, _data), axis = 0)
                self.label = np.concatenate((self.label, _label), axis = 0)
                n = n - 1
            top[0].reshape(*self.data.shape)
            top[1].reshape(*self.label.shape)


        print self.data.shape
        print self.label.shape


    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        if self.random:
            self.idx = random.randint(0, len(self.labelindices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.labelindices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass

    def load_video_and_label(self, idx):
        label = self.load_label(idx)
        label = label[np.newaxis, ...]
        video = self.load_video(idx)
        video = video[np.newaxis, ...]
        return video, label

    def load_video(self, idx):
        images = self.generate_16_images(idx)
        results = np.zeros((48, images[0].shape[1], images[0].shape[2]), np.float32)
        for i in range(16):
            results[i*3+0] = images[i][0,:,:]
            results[i*3+1] = images[i][1,:,:]
            results[i*3+2] = images[i][2,:,:]
        return results

    def generate_16_images(self, idx):
        results = [None] * 16
        im = Image.open(self.imageindices[idx])
        #print str(idx) + '_' + self.imageindices[idx]
        if self.new_W == 0 or self.new_H == 0:
            raise Exception("error parameters input new_W or new_H...")

        im = im.resize((self.new_W+32, self.new_H+32))
        
        mode_gen = random.randint(0, 6) # 0,1,2,3: translation, 4,5: zoom
        results = self.generate_images(im, mode_gen)

        return results

    def generate_images(self, im, mode_gen):
        results = [None] * 16
        width = im.size[0]
        height = im.size[1]
        if mode_gen == 0: # translation from left to center
            for i in range(16):
                results[i] = im.crop((i, 16, i + self.new_W, 16 + self.new_H))
        elif mode_gen == 1: # translation from right to center
            for i in range(16): 
                results[i] = im.crop((width-1-i-self.new_W, 16, width-1-i, 16 + self.new_H))
        elif mode_gen == 2: # translation from top to center
            for i in range(16):
                results[i] = im.crop((16, i, 16 + self.new_W, i + self.new_H))
        elif mode_gen == 3: # translation from bottom to center
            for i in range(16):
                results[i] = im.crop((16, height-1-i-self.new_H, 16 + self.new_W, height-1-i))
        else: # zoom from far to close
            for i in range(16):
                _im = im.crop((i,i,width-i,height-i))
                results[i] = _im.resize((self.new_W, self.new_H))
                
        for i in range(16):
            #results[i].save('./test/results/' + str(mode_gen) + '_' + str(i) + '.png')
            in_ = np.array(results[i], dtype=np.float32)
            in_ = in_[:,:,::-1]
            in_ -= self.mean
            in_ = in_.transpose((2,0,1))
            results[i] = in_

        return results


    def load_label(self, idx):
        label = np.array(self.labelindices[idx], dtype=np.int)
        label = label[np.newaxis, ...]
        return label






