import cv2
import numpy as np
import matplotlib.pyplot as plt

from visualise import drawMatches, draw_matching_scene_keypoints
from features import load_features


def search_scene(scene, cups):
    img1 = cv2.imread('images/small_cup_up_1.jpg',0)
    img2 = cv2.imread(scene,0)

    sift = cv2.SIFT()

    small_features = load_features('features/small_up.txt')
    kp1 = []
    for name in small_features:
	kps,ds = small_features[name]
	kp1 += kps
	print len(kp1)
	try:
	    des1 = np.concatenate((des1, ds))
	except NameError:
	    des1 = ds

    #kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print len(des1)
    print len(des2)

    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)   
    matches = sorted(matches, key = lambda x:x.distance)
    """

    MIN_MATCH_COUNT = 10

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    I2 = img2

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
	if m.distance < 0.7*n.distance:
	    good.append(m)
    #good = [m for m,n in matches]

    if len(good)>MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()

	h,w = img1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)

	cv2.polylines(img2,[np.int32(dst)],True,255,3)

    else:
	print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
	matchesMask = None

    matches = [m for i,m in enumerate(good) if 1 or matchesMask[i]]
    #img3 = overlay_scene_keypoints(img2, kp2, matches)
    img3 = cv2.drawKeypoints(img2, kp2)
    cv2.imwrite("hello.jpg", img3)
    #img3 = drawMatches(img1,kp1,img2,kp2,good)
