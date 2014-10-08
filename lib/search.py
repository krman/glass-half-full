import itertools
import copy

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


def calculate_disparity_map(imgL, imgR):
    stereo = cv2.StereoSGBM(
        minDisparity=64,
        numDisparities=176,
        SADWindowSize=7,
        fullDP=True
    )
    disparity = stereo.compute(imgL, imgR)
    return disparity


def test_polygon(poly):
    """ Return True if bounding polygon is reasonable, else False """

    # Reasonable area?
    lines = [(p1,p2) for p1,p2 in pairwise(poly)]
    lines.append((poly[-1], poly[0])) # join end to start

    area = 0
    for p1,p2 in lines:
        x1,y1 = p1[0][0], p1[0][1]
        x2,y2 = p2[0][0], p2[0][1]
        h = (y1 + y2) / 2
        w = x2 - x1
        area += h * w

    if area < 10000 or area > 100000:
        return False

    # Pretty close to rectangular?
    close = [((0,0),(1,0)), ((0,1),(3,1)), ((1,1),(2,1)), ((2,0),(3,0))]
    for a,b in close:
        ar,ac = a
        br,bc = b
        c = abs(poly[ar][0][ac] - poly[br][0][bc])
        if c > 60:
            return False

    # Similar opposite gradients?
    for a,b in [(0,2), (1,3)]:
        la, lb = lines[a], lines[b]
        x1a, y1a = lines[a][0][0]
        x2a, y2a = lines[a][1][0]
        x1b, y1b = lines[b][0][0]
        x2b, y2b = lines[b][1][0]
        va = [x2a-x1a, y2a-y1a]
        vb = [x1b-x2b, y1b-y2b]
        cos = np.dot(va, vb) / np.linalg.norm(va) / np.linalg.norm(vb)
        deg = np.rad2deg(np.arccos(cos))
        if deg > 20:
            return False
    
    return True


def mask_image(scene, already_found):
    cv2.fillPoly(scene["img"], already_found, (0,0,0))

    delete = []
    for i,kp in enumerate(scene["kp"]):
        for polygon in already_found:
            if cv2.pointPolygonTest(polygon, kp.pt, False) > 0:
                delete.append(i)

    scene["kp"] = np.delete(scene["kp"], delete, axis=0)
    scene["des"] = np.delete(scene["des"], delete, axis=0)

    return scene


def find_matches(features, des2):
    # Find matches in scene
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

    # Sort images by most good matches
    matched.sort(key=lambda x: len(x[1]), reverse=True)
    return matched


def find_homography(matched, features, kp2):
    for name, matches in matched:
        kp1, des1 = features[name]
        img1 = cv2.imread(name, 0)

        # Find homography for this image
        if len(matches) < 4:
            break

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
        dst = dst.astype(np.int32)

        # Reject bad polygons (too big, too small, not approximately rectangle)
        if not test_polygon(dst):
            continue

        # Match is acceptable
        matches = [m for i,m in enumerate(matches) if matchesMask[i]]
        return True, dst, matches

    return False, None, []


def locate_cup(scene, prefix, already_found=[]):
    print prefix

    # Load saved features
    try:
        filename = "features/{0}.txt".format(prefix)
        features = load_features(filename)
    except IOError:
        return already_found

    # Get scene features, but hide features that have already been found
    scene = mask_image(scene, already_found)
    img2 = scene["img"]
    kp2, des2 = scene["kp"], scene["des"]
    if not len(kp2):
        return already_found

    # Find matches and a decent homography
    matched = find_matches(features, des2)
    found, dst, matches = find_homography(matched, features, kp2)
    if not found:
        return already_found

    # ... unless it's already been found
    exists = False
    for polygon in already_found:
        if (dst == polygon).all():
            return already_found

    return locate_cup(scene, prefix, already_found + [dst])


def locate_frame(filename):
    img1 = cv2.imread("images/frame.jpg", 0)
    kp1, des1 = SIFT.detectAndCompute(img1, None)

    img2 = cv2.imread(filename, 0)
    kp2, des2 = SIFT.detectAndCompute(img2, None)

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

    # Find suitable homography
    src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    dst = dst.astype(np.int32)

    # Match is acceptable
    matches = [m for i,m in enumerate(good) if matchesMask[i]]
    draw_matches(img1, kp1, img2, kp2, good)
    cv2.polylines(img2, [np.int32(dst)], True, 230, 3)
    cv2.imshow("frame", img2)
    cv2.waitKey()


def search_scene(filename, disparity, calib):
    found_cups = []
    matrix = calib["intrinsicL"]
    focal_length = matrix[1][1]
    map1, map2 = calib["mapsL"]

    #locate_frame(filename)

    # Locate cups and glasses in image space
    for size in ["small", "medium", "large", "flute"]:
        for orientation in ["up", "down"]:
            for fill in ["empty", "half", "full"]:
                if fill == "half" and "cup" != "flute" \
                        or orientation == "down" and fill != "empty":
                    continue

                # Find features in scene image
                img = cv2.imread(filename, 0)
                img = rectify_image(map1, map2, img)
                kp, des = SIFT.detectAndCompute(img, None)
                scene = {"img":img, "kp":kp, "des":des}

                # Find this kind of cup in the scene
                prefix = "{0}_{1}_{2}".format(size, orientation, fill)
                dst = locate_cup(scene, prefix)
                if not dst:
                    continue
                print "found some!", len(dst)

                cup = {
                    "type": size,
                    "location": (1,1,1),
                    "dst": dst[0],
                    "fill": fill,
                    "orientation": orientation
                }
                found_cups.append(cup)

    # Sort out double bookings
    # maybe do another search, but mask out found cups
    # ie get every cup type to return every homography it finds
    # then sort them out that way with heuristic rules and stuff

    # Mark up image (and calculate positions given depth)
    img = cv2.imread(filename, 0)
    img = rectify_image(map1, map2, img)
    h, w = img.shape
    Ch, Cw = h/2, w/2   # centre
    cv2.circle(img, (Cw,Ch), 5, 0, 3)

    f = focal_length
    print [(cup["type"], cup["fill"], cup["orientation"]) for cup in found_cups]
    for cup in found_cups:
        dst = cup["dst"]
        cv2.polylines(img, [np.int32(dst)], True, 255, 3)
        label = "{0} {1} {2}".format(
                cup["type"], cup["orientation"], cup["fill"])
        plt.text(dst[0][0][0], dst[0][0][1], label, color="white",
                bbox=dict(facecolor='red', alpha=0.8))

    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.show()

    for cup in found_cups:
        text = "{type} cup / {fill} / {orientation}".format(**cup)
        print text

        depth = int(raw_input("depth to camera R (mm)? "))
        (a,b),(c,d),(e,f),(g,h) = dst[0][0], dst[1][0], dst[2][0], dst[3][0]
        ch = (b + d + f + h) / 4
        cw = (a + c + e + g) / 4
        cv2.circle(img, (cw,ch), 5, 255, 3)

        ih, iw = ch - Ch, cw - Cw
        print "location in image space (pixels) (h,w):", (ih,iw)

        rx, ry = iw * depth / f, ih * depth / f
        print "location in real space (mm) (x,y):", (rx,ry)

        label = "{0} {1} {2}".format(
                cup["type"], cup["orientation"], cup["fill"])
        plt.text(dst[0][0][0], dst[0][0][1], label, color="white",
                bbox=dict(facecolor='red', alpha=0.8))
        plt.text(cw, ch, "{0},{1} mm".format(rx,ry), color="white",
                bbox=dict(facecolor='green', alpha=0.8))

    plt.savefig("final.jpg")

    """
    # Find depths of each cup
    for cup in found_cups:
        dst = cup["dst"]
        mask = np.zeros(disparity.shape,np.uint8)
        cv2.drawContours(mask, dst, 0, 255, -1)
        pixelpoints = np.transpose(np.nonzero(mask)).astype(np.int8)
        print cv2.mean(disparity, mask=pixelpoints.astype(np.int8))
    """

    return found_cups, img



if __name__ == '__main__':
    f1 = 'scenes/21_left_cam.jpg'
    f2 = 'scenes/21_right_cam.jpg'
    img1 = cv2.imread(f1, 0)
    img2 = cv2.imread(f2, 0)
    calib = load_calibration('calibration/calib.txt')
    disparity = calculate_disparity_map(img1, img2)
    found, final = search_scene(f2, disparity, calib)

    print found
    plt.imshow(final, cmap=plt.cm.Greys_r)
    plt.show()
