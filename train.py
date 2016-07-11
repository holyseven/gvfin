import caffe
import numpy as np
import C3DImageNetInputLMDBLayer
import C3DImageNetInputLayer
import surgery

# not used
def weightedloss(net, wlayer, border):
    m,k,f,h,w = net.blobs[wlayer].data.shape

    for n in range(k):
        for i in range(h):
            for j in range(w):
                if i - border < 0 or i + border >= h or j - border < 0 or j + border >= w:
                    net.blobs[wlayer].data[0,n,0, i, j] = 0.1
                else:
                    net.blobs[wlayer].data[0,n,0, i, j] = 1



caffe.set_mode_gpu()
caffe.set_device(0)

weights = './c3d_ucf101_fcn.caffemodel'
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)


for _ in range(30):
    solver.step(37500)


