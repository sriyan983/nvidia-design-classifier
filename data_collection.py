#append src folder to read dataset.py from it
import sys
sys.path.append('helpers')

import tkinter
from tkinter import *
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import cv2
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg
import os
from tkinter import messagebox

from dataset import ImageClassificationDataset
import torchvision.transforms as transforms

from reader import read_dataset

global cam
global currFrame
global datasets
global dataset
global countVar
global root

def prepare():
    global datasets
    global dataset

    datasets, dataset = read_dataset()
    
def get_label_selection():
    print("Dataset label: {}".format(value_inside.get()))
    return value_inside.get()

def update_image(change):
    global currFrame
    image = change['new']
    currFrame = image
    
def start():
    global frame
    global cam
    global currFrame
    global countVar
    
    option = get_label_selection()
    if option not in dataset.categories:
        messagebox.showerror('Error', 'Please select a label!')
        return
    
    countVar.set("Image count : {0}".format(dataset.get_count(option)))

    # cam = cv2.VideoCapture(0)
    #cv2.namedWindow("Experience_in_AI camera")
    while cam.running:
        frame = currFrame #cam.read()

        #Update the image to tkinter...
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(np.uint8(frame)).convert('RGB')
        img_update = ImageTk.PhotoImage(PIL_image)
        paneeli_image.configure(image=img_update)
        paneeli_image.image=img_update
        paneeli_image.update()

        if frame.shape[0] == 0:
            print("failed to grab frame")
            break

def stop():
    global cam
    global root
    
    cam.running = False
    cam.unobserve(update_image, names='value')
    root.quit()
    print("Stopped!")
    
def save(event=None):
    global dataset
    global cam
    
    option = get_label_selection()
        
    print("save called")
    dataset.save_entry(cam.value, get_label_selection())
    countVar.set("Image count : {0}".format(dataset.get_count(option)))
    
def on_select(event):
    print("on_select called")
    option = get_label_selection()
    countVar.set("Image count : {0}".format(dataset.get_count(option)))

# create_dataset_folder()

prepare()

cam = USBCamera(width=224, height=224, capture_width=640, capture_height=480, capture_device=0)
image = cam.read()

cam.running = True
cam.observe(update_image, names='value')
    
print(image.shape)

root=tkinter.Tk()
root.title("Dataset creator app")

countVar = StringVar()
countVar.set("Image count : {0}".format("-"))

frame=np.random.randint(0,255,[100,100,3],dtype='uint8')
img = ImageTk.PhotoImage(Image.fromarray(frame))

paneeli_image=tkinter.Label(root) #,image=img)
paneeli_image.grid(row=0,column=0,columnspan=3,pady=1,padx=10)
  
# Variable to keep track of the option
# selected in OptionMenu
value_inside = tkinter.StringVar(root)
  
# Set the default value of the variable
value_inside.set("Select a Label")
  
# Create the optionmenu widget and passing 
# the options_list and value_inside to it.
question_menu = tkinter.OptionMenu(root, value_inside, *dataset.categories, command = on_select)
question_menu.grid(row=1,column=1,pady=1,padx=10)

component_height=5
startButton=tkinter.Button(root,text="Start",command=start,height=5,width=20)
startButton.grid(row=1,column=0,pady=10,padx=10)
startButton.config(height=1*component_height,width=10)

component_height=5
stopButton=tkinter.Button(root,text="Exit",command=stop,height=5,width=20)
stopButton.grid(row=1,column=2,pady=10,padx=10)
stopButton.config(height=1*component_height,width=10)

label = Label(root, textvariable=countVar)
font=('Calibri 14 bold')
label.grid(row=2,column=1,pady=10,padx=10)

label = Label(root, text="Use spacebar to save a snapshot")
font=('Calibri 12 bold')
label.grid(row=3,column=1,pady=10,padx=10)

root.bind("<space>", save)

root.mainloop()
