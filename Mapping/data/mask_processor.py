import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2, os
import numpy as np

masks_list = os.listdir('masks')
masks_list.sort()

def show_image(img, title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color2hex(color):
    r, g, b = color
    return chr(r) + chr(g) + chr(b)

def get_opened_mask(mask, c):
    clr = np.array(c, dtype='uint8')
    bim = cv2.inRange(mask, clr, clr)
    return cv2.morphologyEx(bim, cv2.MORPH_OPEN, None)

def process_color_segment(mask, c):
    color = np.array(c, dtype='uint8')
    # print(color)

    binary_segment = cv2.inRange(mask, color, color)

    mask[np.all(mask == c, axis=-1)] = (0, 0, 0)
    
    binary_segment = cv2.morphologyEx(binary_segment, cv2.MORPH_CLOSE, None)
    binary_segment = cv2.morphologyEx(binary_segment, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
    binary_segment = cv2.erode(binary_segment, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    # show_image(binary_segment)

    color_segment = np.full(mask.shape, color, dtype='uint8')
    # color_segment = cv2.bitwise_and(color_segment, binary_segment)

    cs = cv2.cvtColor(binary_segment, cv2.COLOR_GRAY2RGB)
    inv_cs = cv2.bitwise_xor(cs, np.array([255, 255, 255], dtype='uint8'))
    re_mask = cv2.bitwise_and(mask, inv_cs)
    ccs = cv2.bitwise_and(color_segment, cs)
    result_mask = cv2.bitwise_or(ccs, re_mask)
    # show_image(result_mask)
    return result_mask

def process_mask(mask):
    processed_colors = []
    # Loop through each pixel and process its color segment
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color = mask[i, j, :]
            
            hex_color = color2hex(color)
            if hex_color in processed_colors:
                continue
            processed_colors.append(hex_color)
            
            mask = process_color_segment(mask, color)

    return mask

for msk in masks_list:
    masked_image = cv2.imread(os.path.join('masks', msk), -1)
    refined_mask = process_mask(masked_image)
    show_image(refined_mask)