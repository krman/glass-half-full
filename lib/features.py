#!/usr/bin/python

import cPickle as pickle
import glob

import cv2


SIFT = cv2.SIFT()


def find_features(filename):
    img = cv2.imread(filename,0)
    kps,ds = SIFT.detectAndCompute(img,None)
    return img,kps,ds


def find_all_features(prefix):
    pattern = prefix + '*.jpg'
    for name in sorted(glob.glob(pattern)):
	img,kps,ds = find_features(name)
	yield name,img,kps,ds


def save_features(filename, features):
    """ features is a list of (kps,ds) pairs """

    serialised = []
    for kps,ds in features:
	for i,kp in enumerate(kps):
	    d = ds[i]
	    serialised.append((kp.pt, kp.size, kp.angle, kp.response, 
		kp.octave, kp.class_id, d))

    with open(filename, 'wb') as f:
	pickle.dump(serialised, f)


def load_features(filename):
    with open(filename, 'r') as f:
	pass


if __name__ == '__main__':
    filenames,features = find_all_features('small_cup')
    save_features('small_features', features)
