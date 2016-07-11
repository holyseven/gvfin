import caffe
import C3DSegLayer
import numpy as np
import random as rd
from PIL import Image
import os

def rdColorImage(grey_img_array):
    rd.seed(555) # each time it generates the same color distribution
    colorlist = np.zeros((34,3), dtype=np.uint8)
    for i in range(34):
        colorlist[i] = [rd.randint(0,255), rd.randint(0,255), rd.randint(0,255)]
    color_image = np.zeros((grey_img_array.shape[1], grey_img_array.shape[2], 3), dtype=np.uint8)
    for r in range(grey_img_array.shape[1]):
        for c in range(grey_img_array.shape[2]):
            color_image[r,c] = colorlist[grey_img_array[0,r,c]]
    return color_image

caffe.set_mode_gpu()
caffe.set_device(0)

resultsfilename = './accuracies.txt'
f = open(resultsfilename, 'w')

for iter in range(3000, 60000, 3000):
    fname = '../snapshot/train_iter_' + str(iter) + '.caffemodel'
    print fname
    if os.path.isfile(fname) == False:
        continue
    net = caffe.Net('./fcn_c3d_deploy.prototxt', fname, caffe.TEST)
    results = 0.0

    n = 500
    for i in range(n):
        net.forward()
        out_image = net.blobs['score'].data[0].argmax(axis=0)
        out_image = np.array(out_image, np.uint8)
        color_image = rdColorImage(out_image)
        cimg = Image.fromarray(color_image, 'RGB')
        cimg.save('./results/' + str(iter) + '_' + str(i)+'.png')
    
    
        out = net.blobs['accuracy'].data
        print out
        results = results + out
        if i % 100 == 0 and i > 0:
            print "-------------------"
            print "accuracy: " + str(results/i)
            print "-------------------"
    
    print "-----------results----------"
    print "accuracy: " + str(results/n) #13320,96,500
    f.write(str(iter) + ' caffemodel: \n' )
    f.write('accuracy: ' + str(results/n) + '\n')
    f.flush()

f.close()
