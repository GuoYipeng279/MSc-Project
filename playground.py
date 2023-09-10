import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from humor.test.test_humor import parse_args
from skeleton import Skeleton, ploting, navigation_controller
from main import init_pose, graph
from torch import tensor

args = parse_args(['@./configs/test_humor_sampling.cfg'])
env = Skeleton(args, init_pose, controller=navigation_controller, representer='RELA5')
env.reset()

p = 0

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
    # env.reset()
    for t in range(step):
        state, _,_,_ = env.step(amplitude)
        color = ['blue', 'orange', 'red']
        ax2.scatter(state[0][0], state[0][1],c=color[env.bad])
    joint = env.cur_input_dict['joints'].reshape(22,3).cpu().detach()
    for i in range(joint.shape[0]):
        for c in graph[i]:
            ax1.plot([joint[i,p],joint[c,p]],[joint[i,2],joint[c,2]], c='black')
    ax1.scatter(joint[:,p],joint[:,2])
    canvas.draw()

# def update_plot(_=None, key=None):  # Accept a parameter (ignored)
#     # Update the plot based on slider values
#     if key is None: 
#         plt.scatter(env.state[0][0], env.state[0][1])
#         canvas.draw()
#         return
#     if key is not None:
#         state, _,_,_ = env.step(key)
#         plt.scatter(state[0][0], state[0][1])
#         canvas.draw()

def reset():
    '''
    Reset the playground, so that the agent go back to the reborn point with origin pose
    '''
    env.reset(save=False)
    # plt.cla()
    ax1.cla()
    ax2.cla()
    ax1.axis('equal')
    ax2.axis('equal')
    update_plot()
    canvas.draw()

def close_window():
    root.destroy()

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
reset_button = ttk.Button(root, text='Reset', command=reset)
reset_button.pack()
generate_button = ttk.Button(root, text='Generate', command=update_plot)
generate_button.pack()
close_button = ttk.Button(root, text='Close', command=close_window)
close_button.pack()

# Create matplotlib figure and plot
# fig = plt.figure(figsize=(10,10))
fig, (ax1,ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(20,10))
ax1.axis('equal')
ax2.axis('equal')
# x = np.linspace(0, 2 * np.pi, 100)s
# line, = ax.plot(x, np.sin(x))
# ax1.axis('equal')
# scatter = ax1.scatter(np.array([0,1]),np.array([0,1]))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

update_plot()  # Initial plot update

root.mainloop()
