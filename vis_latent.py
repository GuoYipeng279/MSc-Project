import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from humor.test.test_humor import parse_args
from skeleton import Skeleton, ploting
from main import init_pose
from torch import tensor

args = parse_args(['@./configs/test_humor_sampling.cfg'])
env = Skeleton(args, init_pose)
env.strength = 1
user_data = [0.]*48

def update_plot(_=None):  # Accept a parameter (ignored)
    # Update the plot based on slider values
    amplitude = dimension_input.get()
    if amplitude == '': amplitude = 0
    else: amplitude = int(amplitude)
    step = step_input.get()
    if step == '': step = 1
    else: step = int(step)
    user_data = [0.]*48 # can only test dimensions independently
    user_data[amplitude] = sample_slider.get()
    latent_data = tensor(user_data, device='cuda:0')
    env.reset()
    ans = env.default_roll_out_split(True, latent_data, simu=step)['joints'].reshape((-1,22,3)).cpu().detach().numpy()
    ploting(ans, ax1,ax2)
    canvas.draw()

# Create the main window
root = tk.Tk()
root.title("Real-Time Plot")
input_frame = ttk.Frame(root)
input_frame.pack(padx=10, pady=10)

# Create sliders
dimension_input = ttk.Entry(input_frame)
dimension_input.grid(row=0, column=1)
step_input = ttk.Entry(input_frame)
step_input.grid(row=1, column=1)
sample_slider = ttk.Scale(root, from_=-2, to=2, orient="horizontal")#, command=update_plot)
sample_slider.pack()
generate_button = ttk.Button(root, text='Generate', command=update_plot)
generate_button.pack()

# Create matplotlib figure and plot
fig, (ax1,ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(20,10))
# x = np.linspace(0, 2 * np.pi, 100)s
# line, = ax.plot(x, np.sin(x))
# ax1.axis('equal')
# scatter = ax1.scatter(np.array([0,1]),np.array([0,1]))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

update_plot()  # Initial plot update

root.mainloop()
