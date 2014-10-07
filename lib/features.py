#!/usr/bin/python

import cPickle as pickle
import glob

import cv2


SIFT = cv2.SIFT()


def find_features(filename):
    img = cv2.imread(filename,0)
    kps,ds = SIFT.detectAndCompute(img,None)
    return kps,ds


def find_all_features(prefix):
    pattern = prefix + '*.jpg'
    for name in sorted(glob.glob(pattern)):
        kps,ds = find_features(name)
        yield name,kps,ds


def save_features(filename, features):
    """ features is a list of (name,kps,ds) pairs """

    serialised = {}
    for name,kps,ds in features:
        kps_serial = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response,
                kp.octave, kp.class_id) for kp in kps]
        serialised[name] = (kps_serial,ds)

    with open(filename, 'wb') as f:
        pickle.dump(serialised, f)


def load_features(filename):
    with open(filename, 'r') as f:
        features = pickle.load(f)
        for name in features:
            kps_serial,ds = features[name]
            kps = [cv2.KeyPoint(*kp) for kp in kps_serial]
            features[name] = (kps,ds)

    return features


if __name__ == '__main__':
    features = list(find_all_features('images/small_cup'))
    save_features('small_features', features)
    print load_features('small_features')
