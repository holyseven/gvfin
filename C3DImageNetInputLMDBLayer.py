import caffe

import numpy as np
from PIL import Image

import random
import lmdb

class C3DImageNetInputLMDBLayer(caffe.Layer):
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
        - lmdb_addr
        
        """
        # config
        params = eval(self.param_str)
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.new_H = params.get('new_H', 112) # 112+32
        self.new_W = params.get('new_W', 112) # 112+32
        self.batch_size = params.get('batch_size', 1)
        self.lmdb_addr = params['lmdb_addr']

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        self.idx = 0

        self.lmdb_env = lmdb.open(self.lmdb_addr)
        self.lmdb_txn = self.lmdb_env.begin()
        self.lmdb_cursor = self.lmdb_txn.cursor()
        self.datum = caffe.proto.caffe_pb2.Datum()
        for _ in range(37500*32):
            self.lmdb_cursor.next()
            if _ % 10 == 0:
                print _

    def reshape(self, bottom, top):
        n = self.batch_size
        self.data, self.label = self.load_video_and_label(self.idx)
        n = n - 1
        while n > 0:
            #self.idx = random.randint(0, len(self.labelindices)-1)
            _data, _label = self.load_video_and_label(self.idx)
            self.data = np.concatenate((self.data, _data), axis = 0)
            self.label = np.concatenate((self.label, _label), axis = 0)
            n = n - 1
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)


        #print self.data.shape
        #print self.label.shape


    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        if self.random:
            pass
            #self.idx = random.randint(0, len(self.labelindices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.labelindices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass

    def load_video_and_label(self, idx):
        if not self.lmdb_cursor.next():
            self.lmdb_cursor.first()

        key, value = self.lmdb_cursor.item()
        self.datum.ParseFromString(value)
        img_data = caffe.io.datum_to_array(self.datum)
        label = np.array(self.datum.label, np.int)
        label = label[np.newaxis, ...]

        img_data = img_data.transpose((1,2,0))
        img_data = img_data[:,:,::-1]
        img = Image.fromarray(img_data, 'RGB')
        img = img.resize((self.new_W+32, self.new_H+32))
        

        mode_gen = random.randint(0,6)  # 0,1,2,3: translation, 4,5: zoom
        #img.save('./test/results/' + str(mode_gen) + '_' + '.png')
        video = self.generate_video(img, mode_gen)

        video = video[np.newaxis, ...]
        return video, label

    def generate_video(self, im, mode_gen):
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

        video = np.zeros((48, results[0].shape[1], results[0].shape[2]), np.float32)
        for i in range(16):
            video[i*3+0] = results[i][0,:,:]
            video[i*3+1] = results[i][1,:,:]
            video[i*3+2] = results[i][2,:,:]

        return video








