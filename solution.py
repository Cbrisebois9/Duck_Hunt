import time
import cv2 as cv
import numpy as np

# read kernels
kernels = np.array([cv.imread('roi_down_beak.png'),
            cv.imread('roi_open_beak.png'),
            cv.imread('roi_up_beak.png'),
            np.flip(cv.imread('roi_down_beak.png'),axis=1),
            np.flip(cv.imread('roi_open_beak.png'),axis=1),
            np.flip(cv.imread('roi_up_beak.png'),axis=1)])

# create padding dimensions
tb_pad = int((kernels[0].shape[0]-1)/2)
lr_pad = int((kernels[0].shape[1]-1)/2)

def GetLocation(move_type, env, current_frame):

    time.sleep(1) #artificial one second processing time

    # save current frame with padding in same format as kernels
    cf_shape = current_frame.transpose((1,0,2)).shape
    cf = cv.copyMakeBorder(cv.cvtColor(current_frame.transpose((1,0,2)), cv.COLOR_RGB2BGR),
                            tb_pad,tb_pad,
                            lr_pad,lr_pad,
                            cv.BORDER_CONSTANT,
                            None,0)

    # produce seperate convolution matrix from current frame with each kernel and locate coordinates of most acurate match
    k = 0
    val = 0
    ind = (0,0)
    for kernel in enumerate(kernels):
        match = cv.matchTemplate(cf,kernel[1],cv.TM_CCORR)
        min_val,max_val,min_ind,max_ind = cv.minMaxLoc(match)
        if max_val >= val:
            val = max_val
            ind = max_ind
            k = kernel[0]

    coordinate = ind

    return [{'coordinate' : coordinate, 'move_type' : 'absolute'}]
