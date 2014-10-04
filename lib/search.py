import itertools

import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibrate import load_calibration, rectify_image
from features import load_features
from visualise import draw_matches, draw_matching_scene_keypoints


SIFT = cv2.SIFT()


def pairwise(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ...
    Source: https://docs.python.org/2/library/itertools.html#recipes
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def test_polygon(polygon):
    """ Return True if bounding polygon is reasonable, else False """

    lines = [(p1,p2) for p1,p2 in pairwise(polygon)]
    lines.append((polygon[-1], polygon[0])) # join end to start

    # Reasonable area?
    area = 0
    for p1,p2 in lines:
	x1,y1 = p1[0][0], p1[0][1]
	x2,y2 = p2[0][0], p2[0][1]
	h = (y1 + y2) / 2
	w = x2 - x1
	area += h * w
    
    if area < 10000 or area > 100000:
	return False

    return True


def calculate_disparity_map(imgL, imgR):
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
	return []

    # Sort images by most good matches
    matched.sort(key=lambda x: len(x[1]), reverse=True)

    # Work through images until acceptable homography is found
    found = False
    for name, matches in matched:
	kp1, des1 = features[name]
	img1 = cv2.imread(name, 0)

	# Find homography for this image
	if len(matches) < 4:
	    return []

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
	found = True
	good = matches
	break

    if not found:
	return []

    matches = [m for i,m in enumerate(good) if matchesMask[i]]
    #img3 = draw_matches(img1, kp1, img2, kp2, matches)

    return [dst]


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
		if fill == "half" and "cup" != "champagne" \
			or orientation == "down" and fill != "empty":
		    continue

		prefix = "{0}_{1}_{2}".format(cup, orientation, fill)
		dst = locate_cup(scene, prefix)
		if not dst:
		    continue

		cup = {
		    "type": cup,
		    "location": (1,1,1),
		    "dst": dst[0],
		    "fill": fill,
		    "orientation": orientation
		}
		found_cups.append(cup)

    # Sort out double bookings
		    
    # Mark up image
    for cup in found_cups:
	dst = cup["dst"]
	cv2.polylines(img, [np.int32(dst)], True, 255, 3)
    
    return found_cups, final_image



if __name__ == '__main__':
    img = cv2.imread('scenes/left12.jpg', 0)
    found, final = search_scene(img, None)
    print found
    plt.imshow(final, cmap=plt.cm.Greys_r)
    plt.show()
