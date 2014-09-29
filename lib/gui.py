import sys
import re

import Tkinter as tk
import tkFileDialog

import matplotlib
from matplotlib.figure import Figure
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import cv2
import lib


matplotlib.use('TkAgg')


class CalibrateMode(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

    def draw(self):
	pass


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
	self.prefix_label = tk.Label(self.prefix_frame, text="Category prefix")

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
	t = arange(0.0,3.0,0.01)
	s = sin(2*pi*t)
	self.axes.plot(t,s)
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
	for name,kps,ds in lib.features.find_all_features(prefix):
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
	lib.features.save_features(filename, features)
	self.save_label.config(text="Saved")


    def load_features(self, event=None):
	self.save_label.config(text="")
	filename = tkFileDialog.askopenfilename(
		initialdir='features', defaultextension='.txt')
	if not filename:
	    return

	features = lib.features.load_features(filename)
	
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

    def draw(self):
	pass



class Workspace(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

	self.calibrate = CalibrateMode(self)
	self.training = TrainingMode(self)
	self.features = FeaturesMode(self)
	self.scene = SceneMode(self)
	self.search = SearchMode(self)
    
	self.mode_label = "calibrate"
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
    root.wm_geometry("1024x576")
    root.minsize(271,336)
    Controller(root)
    tk.mainloop()

