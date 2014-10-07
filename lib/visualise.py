import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compare_keypoints(img1, kp1, img2, kp2, color=(255,0,0)):
    """ Draw stitched-together image, including keypoints """

    # Draw keypoints over each image
    #img1 = cv2.drawKeypoints(img1,kp1,color=color)
    #img2 = cv2.drawKeypoints(img2,kp2,color=color)

    # Stitch images together
    r1 = img1.shape[0]
    r2 = img2.shape[0]
    if r1 < r2:                                                           
        z = np.zeros((r2-r1,img1.shape[1]))
        img1 = np.concatenate((img1, z),axis=0)
    elif rows1 > rows2:                                                         
        img2 = np.concatenate((img2,np.zeros((r1-r2,img2.shape[1],3))),axis=0)
    both = np.concatenate((img1,img2),axis=1)
    offset = img1.shape[1]

    # Plot image underneath, in BGR
    #bgr = np.fliplr(both.reshape(-1,3)).reshape(both.shape)
    bgr = both
    plt.imshow(bgr, cmap=plt.cm.Greys_r)
    return offset


def draw_matches(img1, kp1, img2, kp2, matches):
    """ Draw lines between matching keypoints """

    offset = compare_keypoints(img1, kp1, img2, kp2)

    # Extract and plot (x,y) pairs and transform for second image
    for m in matches:
        x1,y1 = kp1[m.queryIdx].pt
        x2,y2 = kp2[m.trainIdx].pt
        x2 += offset
        plt.plot([x1,x2],[y1,y2],'g-')
        plt.plot([x1,x2],[y1,y2],'ro')

    plt.show()


def drawMatchesKnn(img1, kp1, img2, kp2, matches, matchColor=(0,255,0),
        singlePointColor=(255,0,0), matchesMask=None):
    print matches[0]
    print dir(matches[0][0])
    compareKeypoints(img1, kp1, img2, kp2, color=singlePointColor)


def draw_matching_scene_keypoints(scene, kps, matches):
    plt.imshow(scene)
    for m in matches:
        print m
        x,y = kps[m.trainIdx].pt
        plt.plot(x,y,'ro')
    plt.show()
