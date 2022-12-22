import cv2
import os
import numpy as np
import argparse

def rotate_img(dir, img, rotate_angle):
    oimg = cv2.imread(dir + '/' + img,3)
    img90 = np.rot90(oimg)
    img180 = np.rot90(img90)
    img270 = np.rot90(img180)
    (fn,ex) = os.path.splitext(img)
    cv2.imwrite(dir + '/' + fn + "90" + ex, img90)
    cv2.imwrite(dir + '/' + fn + "180" + ex, img180)
    cv2.imwrite(dir + '/' + fn + "270" + ex, img270)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'this is a description')
    parser.add_argument('--input_dir', '-d', help='the input directory', required=True)
    args = parser.parse_args()
    dirlist = os.listdir(args.input_dir)
    for dir in dirlist:
        imglist = os.listdir(args.input_dir + dir)
        for oimg in imglist:
            rotate_img(args.input_dir + dir, oimg, 90)
