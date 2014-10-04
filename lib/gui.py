import os
import sys
import re

import Tkinter as tk
import ttk
import tkFileDialog

import matplotlib
from matplotlib.figure import Figure
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

import cv2

from lib.calibrate import *
from lib.features import *
from lib.search import *


matplotlib.use('TkAgg')


class CalibrateMode(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

	# Calibration image selection
	self.high_frame = tk.Frame(self)
	self.low_frame = tk.Frame(self)

	self.left_label = tk.Label(self.high_frame, text="Left images:  ")
	self.left_entry = tk.Entry(self.high_frame, width=30)
	self.left_entry.insert(0, "calibration/*left*.jpg")
	self.left_entry.bind('<Return>', self.calibrate)

	self.right_label = tk.Label(self.low_frame, text="Right images:")
	self.right_entry = tk.Entry(self.low_frame, width=30)
	self.right_entry.insert(0, "calibration/*right*.jpg")
	self.right_entry.bind('<Return>', self.calibrate)

	self.images_button = tk.Button(self.low_frame, text="Calibrate",
		command=self.calibrate)
	self.load_button = tk.Button(self.low_frame, 
		text="Load calibration...", command=self.load_calibration)

	# Status (display and text)
	self.status_frame = tk.Frame(self)

	self.fig, self.axes = plt.subplots(2, 2, sharex=True, sharey=True)
	self.fig.subplots_adjust(hspace=0.025, wspace=0.025)
	self.fig.tight_layout()

	self.canvas = FigureCanvasTkAgg(self.fig, master=self.status_frame)
	self.canvas.show()

	self.scrollbar = tk.Scrollbar(self.status_frame)
	self.status = tk.Listbox(self.status_frame, width=40, 
		yscrollcommand=self.scrollbar.set)
	self.status.bind("<<ListboxSelect>>", self.select_image)
	self.scrollbar.config(command=self.status.yview)

	# Save feature controls
	self.save_frame = tk.Frame(self)
	self.save_button = tk.Button(self.save_frame, 
		text="Save calibration...", command=self.save_calibration)


    def draw(self):
	# Pack image selection controls
	self.high_frame.pack(side=tk.TOP, fill=tk.X, pady=(20,0), anchor=tk.N)
	self.low_frame.pack(side=tk.TOP, fill=tk.X, pady=20, anchor=tk.N)
	self.left_label.pack(side=tk.LEFT, padx=(20,0))
	self.left_entry.pack(side=tk.LEFT, padx=20)
	self.right_label.pack(side=tk.LEFT, padx=(20,0))
	self.right_entry.pack(side=tk.LEFT, padx=20)
	self.load_button.pack(side=tk.RIGHT, padx=20)
	self.images_button.pack(side=tk.RIGHT, padx=(0,20))

	# Pack status (display and text)
	self.status_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, 
		anchor=tk.E)
	self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
	self.canvas._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
	self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
	self.status.pack(side=tk.RIGHT, fill=tk.Y, anchor=tk.N)

	# Pack save calibration controls
	self.save_frame.pack(side=tk.TOP, fill=tk.X)
	self.save_button.pack(side=tk.RIGHT, padx=20, pady=20, anchor=tk.N)


    def calibrate(self, event=None):
	filenamesL = self.left_entry.get()
	filenamesR = self.right_entry.get()
	self.status.delete(0, tk.END)
	text = "Using filename glob {0} for left,".format(filenamesL)
	self.status.insert(tk.END, text)
	text = "{0} for right".format(filenamesR)
	self.status.insert(tk.END, text)
	self.status.see(tk.END)

	self.images = []
	count = 0
	for done,status,left,right in calibrate_stereo(filenamesL,filenamesR):
	    if done:
		calib = status

	    elif status:
		nameL, cornersL = left
		nameR, cornersR = right
		text = "OK - {0} and {1}".format(nameL, nameR)
		self.status.insert(tk.END, text)
		self.status.update_idletasks()
		self.status.see(tk.END)
		self.status.itemconfig(tk.END, {'fg':'#393'}) 
		count += 1
		imgL = cv2.imread(nameL)
		imgR = cv2.imread(nameR)
		self.images.append((status,imgL,cornersL,imgR,cornersR))
		self.display_image(len(self.images)-1)

	    else:
		nameL, cornersL = left
		nameR, cornersR = right
		text = "FAILED - {0} and {1}".format(nameL, nameR)
		self.images.append((status,0,0,0,0))
		self.status.insert(tk.END, text)
		self.status.update_idletasks()
		self.status.see(tk.END)
		self.status.itemconfig(tk.END, {'fg':'red'}) 
	
	text = "{0} images processed".format(len(self.images))
	self.status.insert(tk.END, text)
	text = "{0} matching chessboards found".format(count)
	self.status.insert(tk.END, text)

	self.set_calib(calib)


    def display_matrix(self, label, matrix):
	self.status.insert(tk.END, label)
	for row in matrix:
	    self.status.insert(tk.END, str(row))
	self.status.see(tk.END)


    def set_calib(self, calib):
	self.calib = calib
	self.display_matrix("Left intrinsic matrix:", calib["intrinsicL"])
	self.display_matrix("Right intrinsic matrix:", calib["intrinsicR"])
	self.status.insert(tk.END, "Undistort/rectify maps also calculated")
	self.status.see(tk.END)
	

    def select_image(self, event):
	selected = int(self.status.curselection()[0])
	size = self.status.size()

	if selected not in [size-1, size-2, 0]:
	    index = selected - 2
	    self.display_image(index)


    def display_image(self, index):
	status,imgL,cornersL,imgR,cornersR = self.images[index]
	self.axes[0][0].clear()
	self.axes[0][1].clear()
	if status:
	    cv2.drawChessboardCorners(imgL, (8,6), cornersL, True)
	    self.axes[0][0].imshow(imgL)
	    self.axes[0][0].set_axis_off()
	    cv2.drawChessboardCorners(imgR, (8,6), cornersR, True)
	    self.axes[0][1].imshow(imgR)
	    self.axes[0][1].set_axis_off()
	self.canvas.draw()


    def load_calibration(self):
	filename = tkFileDialog.askopenfilename(
		defaultextension='.txt', initialdir='calibration',
		filetypes=[('Plain text','*.txt'), ('All files','*.*')])
	if not filename:
	    return

	short = filename.split(os.path.sep)[-1]
	text = "Loading calibration from {0}".format(short)
	self.status.insert(tk.END, text)
	self.status.see(tk.END)

	matrix = load_calibration(filename)
	self.set_matrix(matrix)


    def save_calibration(self):
	filename = tkFileDialog.asksaveasfilename(defaultextension='.txt',
		initialdir='calibration', initialfile="calib.txt",
		filetypes=[('Plain text','*.txt'), ('All files','*.*')])
	if not filename:
	    return

	save_calibration(filename, self.calib)

	short = filename.split(os.path.sep)[-1]
	text = "Calibration saved as {0}".format(short)
	self.status.insert(tk.END, text)
	self.status.see(tk.END)



class TrainingMode(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

    def draw(self):
	pass



class FeaturesMode(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

	# Category prefix controls
	self.prefix_frame = tk.Frame(self)
	self.prefix_label = tk.Label(self.prefix_frame, text="Category prefix:")

	self.prefix_entry = tk.Entry(self.prefix_frame)
	self.prefix_entry.insert(0, "images/")
	self.prefix_entry.bind('<Return>', self.find_features)

	self.prefix_button = tk.Button(self.prefix_frame, text="Find features",
		command=self.find_features)
	self.load_button = tk.Button(self.prefix_frame, 
		text="Load saved features...", command=self.load_features)

	# Status (display and text)
	self.status_frame = tk.Frame(self)

	self.fig = Figure(figsize=(4,3), dpi=100)
	self.axes = self.fig.add_subplot(111)
	self.canvas = FigureCanvasTkAgg(self.fig, master=self.status_frame)
	self.canvas.show()

	self.scrollbar = tk.Scrollbar(self.status_frame)
	self.status = tk.Listbox(self.status_frame, width=40, 
		yscrollcommand=self.scrollbar.set)
	self.status.bind("<<ListboxSelect>>", self.select_feature)
	self.scrollbar.config(command=self.status.yview)

	# Save feature controls
	self.save_frame = tk.Frame(self)
	self.save_label = tk.Label(self.save_frame, text="")
	self.save_button = tk.Button(self.save_frame, text="Save features...",
		command=self.save_features)

	self.features = []


    def draw(self):
	# Pack extract/load features controls
	self.prefix_frame.pack(side=tk.TOP, fill=tk.X, pady=20, anchor=tk.N)
	self.prefix_label.pack(side=tk.LEFT, padx=(20,0))
	self.prefix_entry.pack(side=tk.LEFT, padx=20, expand=True, fill=tk.X)
	self.load_button.pack(side=tk.RIGHT, padx=20)
	self.prefix_button.pack(side=tk.RIGHT, padx=(0,20))

	# Pack status (display and text)
	self.status_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, 
		anchor=tk.E)
	self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
	self.canvas._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
	self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
	self.status.pack(side=tk.RIGHT, fill=tk.Y, anchor=tk.N)

	# Pack save features controls
	self.save_frame.pack(side=tk.TOP, fill=tk.X)
	self.save_button.pack(side=tk.RIGHT, padx=20, pady=20, anchor=tk.N)
	self.save_label.pack(side=tk.RIGHT)


    def find_features(self, event=None):
	self.save_label.config(text="")
	prefix = self.prefix_entry.get()
	self.status.delete(0, tk.END)
	text = "Using prefix {0}".format(prefix)
	self.status.insert(tk.END, text)

	self.features = []
	count = 0
	for name,kps,ds in find_all_features(prefix):
	    self.status.insert(tk.END, name)
	    self.status.update_idletasks()
	    self.status.see(tk.END)
	    count += len(kps)
	    img = cv2.imread(name,0)
	    self.features.append((img,name,kps,ds))
	    self.display_features(img,kps)
	
	text = "{0} images processed".format(len(self.features))
	self.status.insert(tk.END, text)
	text = "{0} features found".format(count)
	self.status.insert(tk.END, text)
	self.status.see(tk.END)


    def select_feature(self, event):
	selected = int(self.status.curselection()[0])
	size = self.status.size()

	if selected not in [size-1, size-2, 0]:
	    index = selected - 1
	    img,name,kps,ds = self.features[index]
	    self.display_features(img, kps)


    def display_features(self, img, kps):
	self.axes.clear()
	img = cv2.drawKeypoints(img, kps)
	self.axes.imshow(img)
	self.canvas.draw()


    def save_features(self, event=None):
	filename = tkFileDialog.asksaveasfilename(
		initialdir='features', defaultextension='.txt')
	if not filename:
	    return

	features = [(n,k,d) for i,n,k,d in self.features]
	save_features(filename, features)
	text = "Saved as {0}".format(filename.split(os.path.sep)[-1])
	self.status.insert(tk.END, text)


    def load_features(self, event=None):
	self.save_label.config(text="")
	filename = tkFileDialog.askopenfilename(
		initialdir='features', defaultextension='.txt')
	if not filename:
	    return

	features = load_features(filename)
	
	self.status.delete(0, tk.END)
	text = "Loading from {0}".format(filename)
	self.status.insert(tk.END, text)

	self.features = []
	count = 0
	for name in features:
	    kps,ds = features[name]
	    self.status.insert(tk.END, name)
	    self.status.update_idletasks()
	    self.status.see(tk.END)
	    count += len(kps)
	    img = cv2.imread(name,0)
	    self.features.append((img,name,kps,ds))
	
	text = "{0} images processed".format(len(self.features))
	self.status.insert(tk.END, text)
	text = "{0} features found".format(count)
	self.status.insert(tk.END, text)
	self.status.see(tk.END)



class SceneMode(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

    def draw(self):
	pass



class SearchMode(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

	# Category prefix controls
	self.select_frame = tk.Frame(self)
	self.left_button = tk.Button(self.select_frame, 
		text="Left photo...", command=lambda: self.load_image(0))
	self.right_button = tk.Button(self.select_frame, 
		text="Right photo...", command=lambda: self.load_image(1))
	self.calib_button = tk.Button(self.select_frame, 
		text="Calibration...", command=self.load_calib)

	self.levels = ["The First Customer", "The Lunch Hour (Rush)",
		"Happy Hour", "Clean-Up"]
	self.level_label = tk.Label(self.select_frame, text="Level:")
	self.level_box = ttk.Combobox(self.select_frame, values=self.levels)
	self.level_index = -1
	self.level_box.current(0)
	self.find_button = tk.Button(self.select_frame, 
		text="Find cups", command=self.find_cups)

	# Status (display and text)
	self.status_frame = tk.Frame(self)

	self.fig, self.axes = plt.subplots(2, 2, sharex=True, sharey=True)
	self.fig.subplots_adjust(hspace=0.025, wspace=0.025)
	self.fig.tight_layout()

	for i,j in [(0,0),(0,1),(1,0),(1,1)]:
	    self.axes[i][j].set_adjustable('box-forced')

	self.canvas = FigureCanvasTkAgg(self.fig, master=self.status_frame)
	self.canvas.show()

	self.scrollbar = tk.Scrollbar(self.status_frame)
	self.status = tk.Listbox(self.status_frame, width=40, 
		yscrollcommand=self.scrollbar.set)
	self.status.bind("<<ListboxSelect>>", None)
	self.scrollbar.config(command=self.status.yview)

	# Save feature controls
	self.save_frame = tk.Frame(self)
	self.save_image_button = tk.Button(self.save_frame, 
		text="Save final image...", command=self.save_image)
	self.save_text_button = tk.Button(self.save_frame,
		text="Save text summary...", command=self.save_text)


    def draw(self):
	# Pack extract/load features controls
	self.select_frame.pack(side=tk.TOP, fill=tk.X, pady=20, anchor=tk.N)
	self.left_button.pack(side=tk.LEFT, padx=20)
	self.right_button.pack(side=tk.LEFT)
	self.calib_button.pack(side=tk.LEFT, padx=20)
	self.find_button.pack(side=tk.RIGHT, padx=(0,20))
	self.level_box.pack(side=tk.RIGHT, padx=(0,20))
	self.level_label.pack(side=tk.RIGHT, padx=(0,20))

	# Pack status (display and text)
	self.status_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, 
		anchor=tk.E)
	self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
	self.canvas._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
	self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
	self.status.pack(side=tk.RIGHT, fill=tk.Y, anchor=tk.N)

	# Pack save features controls
	self.save_frame.pack(side=tk.TOP, fill=tk.X, pady=20, anchor=tk.N)
	self.save_text_button.pack(side=tk.RIGHT, padx=20)
	self.save_image_button.pack(side=tk.RIGHT)


    def draw_image(self, x, y):
	if x == 0 and y == 0:
	    img = self.left
	elif x == 0 and y == 1:
	    img = self.right
	elif x == 1 and y == 1:
	    img = self.final

	self.axes[x][y].clear()
	self.axes[x][y].imshow(img)
	self.axes[x][y].set_axis_off()
	self.canvas.draw()


    def load_image(self, axis):
	filename = tkFileDialog.askopenfilename(
		initialdir='scenes', defaultextension='.jpg',
		filetypes=[('JPEG image','*.jpg'), ('All files','*.*')])
	if not filename:
	    return

	img = cv2.imread(filename)

	if axis == 0:
	    which = "left"
	    self.left = img
	else:
	    which = "right"
	    self.right = img

	self.draw_image(0, axis)
	short = filename.split(os.path.sep)[-1]
	text = "Using {0} for {1} image".format(short, which)
	self.status.insert(tk.END, text)
	self.status.see(tk.END)


    def load_calib(self):
	filename = tkFileDialog.askopenfilename(
		initialdir='calibration', defaultextension='.txt',
		filetypes=[('Plain text','*.txt'), ('All files','*.*')])
	if not filename:
	    return

	self.calib = load_calibration(filename)

	short = filename.split(os.path.sep)[-1]
	text = "Using {0} for calibration".format(short)
	self.status.insert(tk.END, text)
	self.status.see(tk.END)


    def find_disparity_map(self):
	imgL = cv2.cvtColor(self.left, cv2.COLOR_BGR2GRAY)
	imgR = cv2.cvtColor(self.right, cv2.COLOR_BGR2GRAY)
	disparity = calculate_disparity_map(imgL, imgR)
	norm = 255 / disparity.max()
	self.axes[1][0].clear()
	self.axes[1][0].imshow(disparity)
	self.axes[1][0].set_axis_off()
	self.canvas.draw()


    def rectify_source_images(self):
	map1, map2 = self.calib["mapsL"]
	self.left = rectify_image(map1, map2, self.left)
	self.draw_image(0,0)
	map1, map2 = self.calib["mapsR"]
	self.right = rectify_image(map1, map2, self.right)
	self.draw_image(0,1)


    def list_cups(self):
	for cup in self.cups:
	    text = "{type} cup / {location} / {fill} / {orientation}".format(**cup)
	    self.status.insert(tk.END, text)
	self.status.see(tk.END)


    def find_cups(self):
	self.level_index = self.level_box.current()
	level = self.levels[self.level_index]
	self.status.insert(tk.END, "")
	text = "Starting level {0}: '{1}'".format(self.level_index+1, level)
	self.status.insert(tk.END, text)

	self.rectify_source_images()
	disparity = self.find_disparity_map()
	img = cv2.cvtColor(self.right, cv2.COLOR_BGR2GRAY)
	self.cups, self.final = search_scene(img, disparity)
	
	text = "Found {0} cups".format(len(self.cups))
	self.status.insert(tk.END, text)
	self.draw_image(1,1)
	self.list_cups()


    def save_generic(self, event=None, ext=".txt"):
	i = self.level_index
	if i != -1:
	    suggested = "{0}.{1}".format(self.levels[i].replace(" ", ""), ext)
	else:
	    self.status.insert(tk.END, "No search to save")
	    return None

	return tkFileDialog.asksaveasfilename(initialdir='results', 
		initialfile=suggested, defaultextension=ext)


    def save_image(self, event=None):
	filename = self.save_generic(ext='jpg')
	if filename:
	    short = filename.split(os.path.sep)[-1]
	    text = "Saved image as {0}".format(short)
	    self.status.insert(tk.END, text)


    def save_text(self, event=None):
	filename = self.save_generic(ext='csv')
	if filename:
	    short = filename.split(os.path.sep)[-1]
	    text = "Saved summary as {0}".format(short)
	    self.status.insert(tk.END, text)



class Workspace(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

	self.calibrate = CalibrateMode(self)
	self.training = TrainingMode(self)
	self.features = FeaturesMode(self)
	self.scene = SceneMode(self)
	self.search = SearchMode(self)
    
	self.mode_label = None
	self.mode = self.calibrate
	self.change_mode("calibrate")


    def change_mode(self, label):
	if self.mode_label == label:
	    return
	self.mode_label = label

	modes = {
	    "calibrate": self.calibrate,
	    "training": self.training,
	    "features": self.features,
	    "scene": self.scene,
	    "search": self.search
	}

	self.mode.pack_forget()
	self.mode = modes.get(label, self.calibrate)
	self.mode.draw()
	self.mode.pack(expand=True, fill=tk.BOTH)



class Controller(object):
    def __init__(self, master):
	self.master = master

	self.draw_sidebar()
	self.draw_workspace()


    def draw_sidebar(self):
	sidebar = tk.Frame(self.master, borderwidth=1, relief=tk.SUNKEN)
	sidebar.pack(side=tk.LEFT, fill=tk.BOTH)

	modes = [
	    ("Calibrate cameras", "calibrate"),
	    ("Capture training photos", "training"),
	    ("Extract features", "features"),
	    ("Capture scene photos", "scene"),
	    ("Search scene", "search")
	]

	v = tk.IntVar()
	v.set(4)

	for i,mode in enumerate(modes):
	    label,id = mode
	    b = tk.Radiobutton(sidebar, text=label, variable=v, 
		    command=lambda id=id: self.view.change_mode(id),
		    value=i, indicatoron=0, width=30, height=3)
	    b.pack(side=tk.TOP, fill=tk.X, anchor=tk.N)

	button = tk.Button(master=sidebar, text="Quit", command=self._quit,
		width=30, height=2)
	button.pack(side=tk.BOTTOM, fill=tk.X)

	text = "2014 METR4202 Lab 2\nTeam 10A: K Manning / E Barby"
	label = tk.Label(master=sidebar, text=text)
	label.pack(side=tk.BOTTOM, fill=tk.X, pady=10)


    def draw_workspace(self):
	self.view = Workspace(self.master)
	self.view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


    def _quit(self):
	self.master.quit()
	self.master.destroy()



def start_gui():
    root = tk.Tk()
    root.wm_title("glass-half-full: METR4202 Lab 2")
    root.wm_geometry("1280x720")
    root.minsize(271,336)
    Controller(root)
    tk.mainloop()

