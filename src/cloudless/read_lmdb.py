import sys
sys.path.insert(0,'../../../deeplab-public-ver2/python')

import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2
import csv
# Wei Yang 2015-08-19
# Source
#   Read LevelDB/LMDB
#   ==================
#       http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
#   Plot image
#   ==================
#       http://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
#   Creating LMDB in python
#   ==================
#       http://deepdish.io/2015/04/28/creating-lmdb-in-python/

leveldb_dir = "../../../../datasets/planet_cloudless/leveldb/train_leveldb"

PC_DIR = "../../../../datasets/planet_cloudless/"
OUT_DIR = PC_DIR + "images/"

w_train = csv.writer(open(PC_DIR + "train.csv"), delimiter=" ")

db = leveldb.LevelDB(leveldb_dir)
datum = caffe_pb2.Datum()

img_no = 0
print 'a'
for key, value in db.RangeIter():
    print img_no
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)

    cv2.imsave(OUT_DIR + str(i).zfill(10) + '.jpg', data)

    w_train.writerow([OUT_DIR + str(i).zfill(10) + '.jpg', label])
    img_no += 1    
    
