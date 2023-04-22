import sys
sys.path.append('helpers')

import queue

import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk
import cv2

from dataset import ImageClassificationDataset
from reader import read_dataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import json

import threading
import time
from utils import preprocess

global modelPathTF
global epochTF
global modelPathVar
global model
global device
global dataset
global lossValVar
global accValVar
global trainButton
global progBar
global root
global the_queue
global trainThread

def refresh_data():
        global the_queue
        global root
        
        global accValVar
        global lossValVar
        global progBar
        global modelPathTF

        global trainThread

        #print("thread status - ", trainThread.is_alive())

        if not trainThread.is_alive():
            save_model(modelPathTF.get())
            return
        
        # refresh the GUI with new data from the queue
        while not the_queue.empty():
            key, data = the_queue.get()
            #print("value from queue : {0}, {1}", key, data)
            if key == "prog":
                progBar["value"] = data
            elif key == "loss":
                lossValVar.set(data)
            elif key == "accu":
                accValVar.set(data)

        #  timer to refresh the gui with data from the asyncio thread
        root.after(100, refresh_data)  # called only once!

def start():
    global modelPathTF
    global epochTF
    global trainButton
    global the_queue
    global root
    
    global trainThread
    
    print("---")
    if int(epochTF.get()) <= 0:
        messagebox.showerror('Error', 'Provide an epoch value greater than 0')
        return
    
    trainButton.config(state="disabled")
    
    root.after(1, train_prepare(modelPathTF.get()))
    root.after(100, refresh_data);

    trainThread = threading.Thread(target=train, args=(the_queue, int(epochTF.get()), True))
    trainThread.start()
        
def stop():
    global root
    print("---")
    root.quit()
    print("Stopped!")

def launch():
    global modelPathTF
    global epochTF
    global modelPathVar
    global lossValVar
    global accValVar
    global trainButton
    global progBar
    global root

    root=tkinter.Tk()
    root.title("Train classifier")

    countVar = StringVar()
    countVar.set("Image count : {0}".format("-"))
    
    accValVar = StringVar()
    lossValVar = StringVar()

    valueInside = tkinter.StringVar(root)

    valueInside.set("Select a Label")

    label = Label(root, text="Model Path")
    font=('Calibri 12 bold')
    label.grid(row=0,column=0,pady=10,padx=10)

    modelPathVar = tkinter.StringVar()
    modelPathTF = tkinter.Entry( root, textvariable=modelPathVar)
    modelPathVar.set("models/model_v1.pth")
    # entry1.place(x = 80, y = 50)  
    modelPathTF.grid(row=0,column=1,pady=10,padx=5)

    label = Label(root, text="Epocs")
    font=('Calibri 12 bold')
    label.grid(row=1,column=0,pady=10,padx=10)

    epocValVar = tkinter.StringVar()
    epochTF = Entry(root, textvariable=epocValVar)
    epocValVar.set("0")
    # entry1.place(x = 80, y = 50)  
    epochTF.grid(row=1,column=1,pady=10,padx=5,columnspan=1)

    label = Label(root, text="Epocs")
    font=('Calibri 12 bold')
    label.grid(row=1,column=0,pady=10,padx=10)

    label = Label(root, text="Current epoch")
    font=('Calibri 12 bold')
    label.grid(row=2,column=0,pady=10,padx=10)

    progBar = ttk.Progressbar(root,orient=HORIZONTAL, length=200,mode="determinate")
    progBar.grid(row=2,column=1,pady=10,padx=5)
    progBar['value']=0

    accLabel = Label(root, text="Accuracy")
    font=('Calibri 12 bold')
    accLabel.grid(row=3,column=0,pady=10,padx=10)

    accValLabel = Label(root, textvariable=accValVar)
    accValVar.set("{0}".format("-"))
    font=('Calibri 12 normal')
    accValLabel.grid(row=3,column=1,pady=10,padx=10)

    lossLabel = Label(root, text="Loss")
    font=('Calibri 12 bold')
    lossLabel.grid(row=4,column=0,pady=10,padx=10)

    lossValLabel = Label(root, textvariable=lossValVar)
    lossValVar.set("{0}".format("-"))
    font=('Calibri 12 normal')
    lossValLabel.grid(row=4,column=1,pady=10,padx=10)

    componentHeight=2
    trainButton=tkinter.Button(root,text="Train",command=start,height=5,width=20)
    trainButton.grid(row=5, column=0,sticky='W',pady=10,padx=10)
    trainButton.config(height=1*componentHeight,width=5, padx=50)

    componentHeight=2
    stopButton=tkinter.Button(root,text="Exit",command=stop,height=5,width=20)
    stopButton.grid(row=5, column=1,sticky='E',pady=10,padx=10)
    stopButton.config(height=1*componentHeight,width=5, padx=50)

    root.resizable(0, 0)
    root.mainloop()
    
def load_model(path):
    global model
    model.load_state_dict(torch.load(path))

def save_model(path):
    global model
    parDir = "models"
    isExist = os.path.exists(parDir)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(parDir)
    torch.save(model.state_dict(), path)
    
def train_prepare(modelPath):
    print(type(modelPath), modelPath)
    global model
    global device
    global dataset
    
    time.sleep(1)
    
    datasets, dataset = read_dataset()
    
    print("fc layer output - {0}".format(len(dataset.categories)))
    
    device = torch.device('cuda')
    
    # RESNET 18
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, len(dataset.categories))
    
    model = model.to(device)
    
    isExist = os.path.exists(modelPath)
    if isExist:
        load_model(modelPath)
    else:
        print("model not found in current path, may be this is the first time you are about to train!");

    # display(model_widget)
    print("model configured and model_widget created")

def train(the_queue, epochs, is_training):
    global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, dataset, optimizer
    global device
    
    print("all data types - ", type(the_queue), type(epochs), type(is_training))
    total_epocs = epochs
        
    BATCH_SIZE = 3
    optimizer = torch.optim.Adam(model.parameters())
    elspased_epocs = 0
    
    try:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        time.sleep(1)

        if is_training:
            model = model.train()
        else:
            model = model.eval()
        while total_epocs > 0:
            i = 0
            sum_loss = 0.0
            error_count = 0.0
            for images, labels in iter(train_loader):
                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                if is_training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images)

                # compute loss
                loss = F.cross_entropy(outputs, labels)

                if is_training:
                    # run backpropogation to accumulate gradients
                    loss.backward()

                    # step optimizer to adjust parameters
                    optimizer.step()

                # increment progress
                error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
                count = len(labels.flatten())
                i += count
                sum_loss += float(loss)
                
                # print("loss actual - {0}", sum_loss / i)
                # print("accuracy actual - {0}", 1.0 - error_count / i)

                the_queue.put(("accu", "{0}".format(1.0 - error_count / i)))
                the_queue.put(("loss", "{0}".format(sum_loss / i)))
        
                # lossValVar.set("{0}".format(sum_loss / i))
                # accValVar.set("{0}".format(1.0 - error_count / i))
            
            elspased_epocs = elspased_epocs + 1
            if is_training:
                total_epocs = total_epocs - 1
                print("elspased_epocs", elspased_epocs)
                # epochs_widget.value = epochs_widget.value - 1
                progBar['value'] = elspased_epocs * (100 / epochs)
            else:
                break
    except Exception as ex:
        print(ex)
        pass
    model = model.eval()
    
trainThread = None
the_queue = queue.Queue()

launch()

