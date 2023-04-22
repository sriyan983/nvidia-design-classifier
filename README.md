This project is implementation made as part of NVIDIA's AI Specialist certification. The concept uses an image classification using ResNet18 (through transfer learning) and trains itself to classify different designs of gift boxes of packaging firm of mine. The classification results shall be used for automated printing of appropriate price tags and also assist in packing them as stacks assuring no shuffling to happen! 

The project uses the following,

Environment: 
- Jetson AGX Xavier 
- Jetpack version: 5.1.1 [L4T 35.3.1]
- Cuda version: Cuda compilation tools, release 11.4, V11.4.315
- Torch version : 
>>> print(torch.__version__)
1.14.0a0+44dac51c.nv23.02
- TorchVision version : 
>>> print(torchvision.__version__)
0.14.0a0+5ce4506

Connections:
- USB Camera (Logitech)
- Mouse & Keyboard
- Monitor
- Ethernet cable 
- Data cable to connect the linux host system to install JetPack and boot Jetson AGX Xavier. 

Modules:
- Datacollection module for creating the dataset with a Tkinter based UI for interactivity and works standalone.
- Training module with a Tkinter based UI for interactivity and works standalone.
- Live Inference module with a Tkinter based UI for interactivity and works standalone.

To Run:
1. First run the data collection application to collect data for different labels we have defined
python data_collection.py
2. Then run the training application to train the model through multiple iterations
python train.py
3. Finally run the classiifcation application to infer real time using a camera.
python classify.py

(Edit the json file based on your needs. You can also get the dataset and the model file from https://drive.google.com/drive/u/2/folders/1gvx24EeKoxC14XldWufA7lIufHK3073V)

Notes: 

- All the above applications run standalone and allows you to save different versions of models using Entry widget from Tkinter.
- A json file config.json allows you to configure the tasks, datasets and labels such that you don't need to worry about synchronizing them across these three standalone applications.

Git Repo: https://github.com/sriyan983/nvidia-design-classifier.git

Blog: https://medium.com/@sriyan983/machine-learning-using-jetson-21df58787742

Video: https://youtu.be/G-FMJRb_sVU


