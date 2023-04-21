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
import threading
import time
import queue

from utils import preprocess

from tkinter import messagebox

from dataset import ImageClassificationDataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from reader import read_dataset

global frame
global cam
global currFrame
global classificationVar
global classifyThread
global root
global theQueue

def update_image(change):
    global currFrame
    image = change['new']
    currFrame = image
    
def refresh_data():
    global theQueue
    global root
    global classificationVar
    global classifyThread

    #print("thread status - ", classifyThread.is_alive())

    if not classifyThread.is_alive():
        return

    # refresh the GUI with new data from the queue
    while not theQueue.empty():
        key, data = theQueue.get()
        #print("value from queue : {0}, {1}", key, data)
        if key == "result":
            classificationVar.set(data)

    #  timer to refresh the gui with data from the asyncio thread
    root.after(100, refresh_data)  # called only once!
        
def start():
    global frame
    global cam
    global currFrame
    global classificationVar
    global root
    global classifyThread
    
    startButton.config(state="disabled")

    datasets, dataset = read_dataset()
        
    device = torch.device('cuda')
   
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, len(dataset.categories))

    model = model.to(device)

    model.load_state_dict(torch.load(modelPathTF.get()))
    
    root.after(100, refresh_data);
        
    classifyThread = threading.Thread(target=live, args=((cam, dataset, model, theQueue)))
    classifyThread.start()
    
    while cam.running:
        frame = currFrame #cam.read()

        #Update the image to tkinter...
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(np.uint8(frame)).convert('RGB')
        img_update = ImageTk.PhotoImage(PIL_image)
        imageView.configure(image=img_update)
        imageView.image=img_update
        imageView.update()

        if frame.shape[0] == 0:
            print("failed to grab frame")
            break

def live(camera, dataset, model, theQueue):
    while cam.running:
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        #print("---prediction---")
        
        predictedclass = dataset.categories[category_index]
        #print(predictedclass)
        for i, score in enumerate(list(output)):
            #print(i, score)
            if dataset.categories[i] == predictedclass:
                rounded = round(float(score), 2)
                # theQueue.put(("result", "{0} @ {1}".format(predictedclass, rounded)))
                theQueue.put(("result", "{0}".format(predictedclass)))

        #print("---end---")
            
def stop():
    global cam
    global root
    
    cam.running = False
    cam.unobserve(update_image, names='value')
    root.quit()
    print("Stopped!")

classifyThread = None
theQueue = queue.Queue()
    
cam = USBCamera(width=224, height=224, capture_width=640, capture_height=480, capture_device=0)
image = cam.read()

cam.running = True
cam.observe(update_image, names='value')
    
print(image.shape)

root=tkinter.Tk()
root.title("Classifier app")

frame=np.random.randint(0,255,[100,100,3],dtype='uint8')
img = ImageTk.PhotoImage(Image.fromarray(frame))

imageView=tkinter.Label(root) #,image=img)
imageView.grid(row=0,column=0,columnspan=3,pady=1,padx=10)

modelPathVar = tkinter.StringVar()
modelPathTF = tkinter.Entry( root, textvariable=modelPathVar)
modelPathVar.set("models/model_v1.pth")
# entry1.place(x = 80, y = 50)  
modelPathTF.grid(row=1,column=1,pady=10,padx=5)

label = Label(root, text="Classification result")
font=('Calibri 14 bold')
label.grid(row=2,column=1,pady=0,padx=10)

classificationVar = tkinter.StringVar()
label = Label(root, textvariable=classificationVar)
classificationVar.set("Waiting...")
font=('Calibri 12')
label.config(fg="#0000FF")
label.config(bg="yellow")
label.grid(row=3,column=1,pady=10,padx=10)

component_height=2
startButton=tkinter.Button(root,text="Start",command=start,height=5,width=20)
startButton.grid(row=4,column=0,pady=10,padx=10)
startButton.config(height=1*component_height,width=5)

component_height=2
stopButton=tkinter.Button(root,text="Exit",command=stop,height=5,width=20)
stopButton.grid(row=4,column=2,pady=10,padx=10)
stopButton.config(height=1*component_height,width=5)

root.mainloop()