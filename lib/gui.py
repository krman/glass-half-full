import sys
import re

import Tkinter as tk

import matplotlib
from matplotlib.figure import Figure
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from features import *


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

	# Status (display and text)
	self.status_frame = tk.Frame(self)

	self.fig = Figure(figsize=(5,4), dpi=100)
	self.axes = self.fig.add_subplot(111)
	t = arange(0.0,3.0,0.01)
	s = sin(2*pi*t)
	self.axes.plot(t,s)
	self.canvas = FigureCanvasTkAgg(self.fig, master=self.status_frame)
	self.canvas.show()

	self.scrollbar = tk.Scrollbar(self.status_frame)

	self.status_text = tk.Text(self.status_frame, width=40, 
		yscrollcommand=self.scrollbar.set)
	self.status_text.insert(tk.END, "hello, ")
	self.status_text.insert(tk.END, "world")
	self.status_text.config(state=tk.DISABLED)

	self.scrollbar.config(command=self.status_text.yview)

	# Save feature controls
	self.save_frame = tk.Frame(self)
	self.save_label = tk.Label(self.save_frame, text="Filename to save")
	self.save_entry = tk.Entry(self.save_frame)
	self.save_entry.insert(0, "features/")
	self.save_button = tk.Button(self.save_frame, text="Save features")


    def draw(self):

	# Pack image prefix controls
	self.prefix_frame.pack(side=tk.TOP, fill=tk.X)
	self.prefix_label.pack(side=tk.LEFT, padx=20, pady=20, anchor=tk.N)
	self.prefix_entry.pack(side=tk.LEFT, padx=20, pady=20, 
		anchor=tk.N, expand=True, fill=tk.X)
	self.prefix_button.pack(side=tk.RIGHT, padx=20, pady=20, anchor=tk.N)

	# Pack status (display and text)
	self.status_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, 
		anchor=tk.E)
	self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
	self.canvas._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
	self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
	self.status_text.pack(side=tk.RIGHT, fill=tk.Y, padx=20, anchor=tk.N)

	# Pack save feature controls
	self.save_frame.pack(side=tk.TOP, fill=tk.X)
	self.save_label.pack(side=tk.LEFT, padx=20, pady=20, anchor=tk.N)
	self.save_entry.pack(side=tk.LEFT, padx=20, pady=20, 
		anchor=tk.N, expand=True, fill=tk.X)
	self.save_button.pack(side=tk.RIGHT, padx=20, pady=20, anchor=tk.N)


    def find_features(self, event=None):
	self.status_text.config(state=tk.NORMAL)
	prefix = self.prefix_entry.get()
	self.status_text.delete(1.0, tk.END)
	text = "Using prefix {0}\n".format(prefix)
	self.status_text.insert(tk.END, text)

	features = []
	count = 0
	for name,img,kps,ds in find_all_features(prefix):
	    self.status_text.insert(tk.END, "\n")
	    self.status_text.insert(tk.END, name)
	    self.status_text.update_idletasks()
	    self.status_text.see(tk.END)
	    features.append((kps,ds))
	    count += len(kps)

	text = "\n\n{0} images processed\n{1} features found".format(
		len(features), count)
	self.status_text.insert(tk.END, text)
	self.status_text.see(tk.END)
	self.status_text.config(state=tk.DISABLED)

	self.axes.clear()
	t = arange(0.0,3.0,0.01)
	s = sin(3*pi*t)

	self.axes.plot(t,s)
	self.canvas.draw()

	self.save_entry.delete(0, tk.END)
	suggested = re.sub("^images/", "features/", prefix) + ".txt"
	self.save_entry.insert(tk.END, suggested)
	


class SceneMode(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

    def draw(self):
	pass



class SearchMode(tk.Frame):
    def __init__(self, *args, **kwargs):
	tk.Frame.__init__(self, *args, **kwargs)

    def draw(self):
	print "drawing search"



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

