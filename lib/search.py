import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibrate import load_calibration, rectify_image
from features import load_features
from visualise import draw_matches, draw_matching_scene_keypoints


SIFT = cv2.SIFT()


def test_polygon(polygon):
    for line in polygon:
	print line
    return True


def calculate_disparity_map(imgL, imgR):
    #stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, 0, 15)
    stereo = cv2.StereoSGBM(
	minDisparity=64,
	numDisparities=176,
	SADWindowSize=7,
	fullDP=True
    )
    disparity = stereo.compute(imgL, imgR)
    return disparity


def locate_cup(scene, prefix):
    """ retval is dict of bbox:score """

    # Get scene features
    img2 = scene["img"]
    kp2, des2 = scene["kp"], scene["des"]

    # Load saved features
    try:
	filename = "features/{0}.txt".format(prefix)
	print "locating cups from {0}".format(filename)

	features = load_features(filename)
	kp1 = []
	matched = []
	for name in features:
	    kp1, des1 = features[name]

	    # Match features with FLANN-based matcher
	    FLANN_INDEX_KDTREE = 0
	    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	    search_params = dict(checks = 50)

	    flann = cv2.FlannBasedMatcher(index_params, search_params)
	    matches = flann.knnMatch(des1, des2, k=2)

	    # Only store close-enough matches (Lowe's ratio test)
	    good = []
	    for m,n in matches:
		if m.distance < 0.7*n.distance:
		    good.append(m)

	    # If this is better than the old best for this category, store it
	    if len(good) > 0:
		matched.append((name,good))

    except IOError:
	return

    # Sort images by most good matches
    matched.sort(key=lambda x: len(x[1]), reverse=True)

    # Work through images until acceptable homography is found
    found = False
    for name, matches in matched:
	kp1, des1 = features[name]
	img1 = cv2.imread(name, 0)
	plt.imshow(img1)

	# Find homography for this image
	if len(matches) < 4:
	    return

	src_pts = np.float32(
		[kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
	dst_pts = np.float32(
		[kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3)
	matchesMask = mask.ravel().tolist()

	# Reject upside-down images (matched separately)
	if M[0][0] < 0:
	    continue

	h,w = img1.shape
	pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)

	# Reject bad polygons (too big, too small, not approximately rectangle)
	if not test_polygon(dst):
	    continue

	# Match is acceptable
	cv2.polylines(img2, [np.int32(dst)], True, 255, 3)
	found = True
	good = matches
	break

    if not found:
	return

    matches = [m for i,m in enumerate(good) if matchesMask[i]]
    img3 = draw_matches(img1, kp1, img2, kp2, matches)

    bbox = (0,0,20,30) # x,y,w,h
    return {bbox:20}


def locate_glass(scene, prefix):
    print "glass", prefix


def search_scene(img, disparity):
    found_cups = []
    final_image = img

    # Find features in scene image
    kp, des = SIFT.detectAndCompute(img, None)
    scene = {"img":img, "kp":kp, "des":des}

    # Locate cups and glasses in image space
    for cup in ["small", "medium", "large", "champagne"]:
	for orientation in ["up", "down"]:
	    for fill in ["empty", "half", "full"]:
		prefix = "{0}_{1}_{2}".format(cup, orientation, fill)
		if cup == "champagne":
		    locate_glass(scene, prefix)
		elif fill == "half":
		    continue
		else:
		    locate_cup(scene, prefix)

    return found_cups, final_image


def osearch_scene(img, disparity):
    img1 = cv2.imread('images/small_cup_up_30.jpg',0)
    img2 = img

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
    img3 = draw_matches(img1,kp1,img2,kp2,good)




if __name__ == '__main__':
    img = cv2.imread('scenes/left4.jpg', 0)
    search_scene(img, None)
