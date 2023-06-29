#!/usr/bin/python
#
# Convert instances from png files to a dictionary
# This files is created according to https://github.com/facebookresearch/Detectron/issues/111

from __future__ import print_function, absolute_import, division
import os, sys

# sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from ..helpers.csHelpers import *

# Cityscapes imports
from .instance import *
from ..helpers.csHelpers import *
from ..helpers.labels import *

import cv2
# import lycon
#from create_dataset.utils import cv2_util


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def instances2dict_with_polygons(seg_imageFileList,ins_imageFileList,verbose=False):
    imgCount     = 0
    instanceDict = {}
    #import pdb;pdb.set_trace()
    if not isinstance(seg_imageFileList, list):
        seg_imageFileList = [seg_imageFileList]

    if verbose:
        print("Processing {} images...".format(len(seg_imageFileList)))
        print("Processing {} images...".format(len(ins_imageFileList)))

    for imageFileName_seg,imageFileName_ins in zip(seg_imageFileList,ins_imageFileList):
        print("Segment file:",imageFileName_seg)
        print("Instance files:",imageFileName_ins)
        # img = lycon.load(imageFileName_ins) # (1738, 1956, 3)
        # img_seg = lycon.load(imageFileName_seg) # (1738, 1956, 3) segmentation file
        img = cv2.imread(imageFileName_ins)[...,::-1]
        img_seg = cv2.imread(imageFileName_seg)[...,::-1]

        # Image as numpy array
        imgNp = np.array(img) # Gives h * w * 3 matrix
        imgNp_seg = np.array(img_seg)
        if not (imgNp.ndim and imgNp_seg.ndim) == 3:
            import pdb;pdb.set_trace(); 
        imgNp = imgNp[:,:,0] + 256 * imgNp[:,:,1] + 256*256*imgNp[:,:,2]
        imgNp_seg = imgNp_seg[:,:,0] + 256 * imgNp_seg[:,:,1] + 256*256*imgNp_seg[:,:,2]


        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp): #np.unique(imgNp)
        #for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp,imgNp_seg, instanceId)
            instanceObj_dict = instanceObj.toDict()

            #instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
            # if id2label[instanceObj.labelID].hasInstances:
            if m2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                #contour, hier = cv2_util.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour, hier = findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict['contours'] = polygons

            instances[m2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey= os.path.abspath(imageFileName_ins)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

# def main(argv):
#     seg_fileList = []
#     ins_fileList = []
#     if (len(argv) > 2):
#         for arg in argv:
#             if ("png" in arg):
#                 seg_fileList.append(arg)
#                 ins_fileList.append(arg)
#     instances2dict_with_polygons(seg_fileList,ins_fileList, True)

# if __name__ == "__main__":
#     main(sys.argv[1:])