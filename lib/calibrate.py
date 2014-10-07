#!/usr/bin/python

import glob
import cPickle as pickle

import cv2
import numpy as np

from matplotlib import pyplot as plt


SIFT = cv2.SIFT


def calibrate_stereo(left_filenames, right_filenames):
    left_images = sorted(glob.glob(left_filenames))
    right_images = sorted(glob.glob(right_filenames))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

    obj_points = []
    img_pointsL = []
    img_pointsR = []

    for i in range(len(left_images)):
        nameL = left_images[i]
        nameR = right_images[i]

        imgL = cv2.imread(nameL, 0)
        imgR = cv2.imread(nameR, 0)

        retL, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
        retR, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)

        if retL and retR:
            obj_points.append(objp)
            cv2.cornerSubPix(imgL, cornersL, (11,11), (-1,-1), criteria)
            img_pointsL.append(cornersL)
            cv2.cornerSubPix(imgR, cornersR, (11,11), (-1,-1), criteria)
            img_pointsR.append(cornersR)
            yield False, True, (nameL,cornersL), (nameR,cornersR)
        else:
            yield False, False, (nameL,None), (nameR,None)

    image_size = imgL.shape[::-1]

    # Find individual intrinsic parameters first
    retL, matrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
            obj_points, img_pointsL, image_size, None, None)
    retR, matrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
            obj_points, img_pointsR, image_size, None, None)

    # Calibrate stereo camera, with calculated intrinsic parameters
    ret, matrixL, distL, matrixR, distR, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_pointsL, img_pointsR, image_size,
            matrixL, distL, matrixR, distR, flags=cv2.CALIB_FIX_INTRINSIC)

    # Calculate rectification transforms for each camera
    R1 = np.zeros((3,3), np.float32)
    R2 = np.zeros((3,3), np.float32)
    P1 = np.zeros((3,4), np.float32)
    P2 = np.zeros((3,4), np.float32)
    Q = np.zeros((4,4), np.float32)
    R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
            matrixL, distL, matrixR, distR, image_size, R, T)

    # Create undistortion/rectification map for each camera
    mapsL = cv2.initUndistortRectifyMap(matrixL, distL, R1, P1,
            image_size, cv2.CV_32FC1)
    mapsR = cv2.initUndistortRectifyMap(matrixR, distR, R2, P2,
            image_size, cv2.CV_32FC1)

    # Convert maps to fixed-point representation (faster)
    mapsL = cv2.convertMaps(mapsL[0], mapsL[1], cv2.CV_32FC1)
    mapsR = cv2.convertMaps(mapsR[0], mapsR[1], cv2.CV_32FC1)

    calib = {
        "intrinsicL": matrixL, "intrinsicR": matrixR,
        "mapsL": mapsL, "mapsR": mapsR,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "roiL": roiL, "roiR": roiR
    }
    yield True, calib, None, None


def rectify_image(map1, map2, image):
    return cv2.remap(image, map1, map2, cv2.INTER_NEAREST)


def get_fundamental_matrix(left_image, right_image):
    img1 = cv2.imread(left_image, 0)
    img2 = cv2.imread(right_image, 0)

    kp1, des1 = SIFT.detectAndCompute(img1,None)
    kp2, des2 = SIFT.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.array(pts1, dtype='float32')
    pts2 = np.array(pts2, dtype='float32')

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    return F


def save_calibration(filename, calib):
    with open(filename, 'wb') as f:
        pickle.dump(calib, f)


def load_calibration(filename):
    with open(filename, 'r') as f:
        matrix = pickle.load(f)
    return matrix



if __name__ == '__main__':
    #matrix = np.array([])
    #for done,mtx,r,c in calibrate_camera('calibration/*left*'):
        #if done:
            #matrix = mtx
    #save_calibration('calibration/left.txt', matrix)
    #print load_calibration('calibration/left.txt')

    for a in calibrate_stereo('calibration/*left*.jpg', 'calibration/*right*.jpg'):
        print a[0]
        if a[0]:
            print a[1]
