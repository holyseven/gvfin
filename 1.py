import caffe
import numpy as np
import C3DSegLayer
import surgery




caffe.set_mode_gpu()
caffe.set_device(0)

weights = './c3d_ucf101_fcn.caffemodel'
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

solver.step(2)
solver.step(2)



import caffe
import lmdb

lmdb_env = lmdb.open('./')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)


for ( idx, (key, value) ) in enumerate(cursor): 


http://chrischoy.github.io/research/reading-protobuf-db-in-python/
