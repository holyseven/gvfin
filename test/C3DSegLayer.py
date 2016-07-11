import caffe

import numpy as np
from PIL import Image

import random


class C3DSegLayer(caffe.Layer):
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
        
        """
        # config
        params = eval(self.param_str)
        self.imagelist = params['imagelist']
        self.labellist = params['labellist']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.new_H = params.get('new_H', 0)
        self.new_W = params.get('new_W', 0)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        # load address of images and labels
        _allimageindices = open(self.imagelist, 'r').read().splitlines()
        self.labelindices = open(self.labellist, 'r').read().splitlines()
        if len(_allimageindices) != 30 * len(self.labelindices):
            raise Exception("Images and labels should pair.")

        self.imageindices = [None]*len(self.labelindices)
        for i in range(len(self.labelindices)):
            self.imageindices[i] = [None] * 30
            for j in range(30):
                self.imageindices[i][j] = _allimageindices[i*30+j]

        self.idx = 0
        # random
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.labelindices)-1)

    def reshape(self, bottom, top):
        self.data = self.load_video(self.idx)
        self.label = self.load_label(self.idx)

        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

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

    def load_video(self, idx):
        images = self.load_16_images(idx)
        results = np.zeros((48, images[0].shape[1], images[0].shape[2]), np.float32)
        for i in range(16):
            results[i*3+0] = images[i][0,:,:]
            results[i*3+1] = images[i][1,:,:]
            results[i*3+2] = images[i][2,:,:]
        return results


    def load_16_images(self, idx):
        results = [None] * 16
        for i in range(16):
            im = Image.open(self.imageindices[idx][19-i])
            #if i == 0:
            #    print "loading image: " + self.imageindices[idx][19]
            if self.new_W != 0 and self.new_H !=0:
                im = im.resize((self.new_W, self.new_H))
            in_ = np.array(im, dtype=np.float32)
            in_ = in_[:,:,::-1]
            in_ -= self.mean
            in_ = in_.transpose((2,0,1))
            results[i] = in_
        return results

    def load_label(self, idx):
        im = Image.open(self.labelindices[idx])
        #print "loading label image: " + self.labelindices[idx]
        if self.new_W != 0 and self.new_H !=0:
            im = im.resize((self.new_W, self.new_H))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label



