#!/usr/bin/python

import sys
import argparse

from lib.search import search_scene
from lib.gui import start_gui


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="scene",
	    help="scene image filename")
    parser.add_argument("-c", dest="cups", action='append',
	    choices=['all','small','medium','large','champagne'],
	    help="cups to find in scene")
    args = parser.parse_args()

    if not args.cups or 'all' in args.cups:
	args.cups = ['all']

    if args.scene:
	search_scene(args.scene, args.cups)
    else:
	start_gui()
