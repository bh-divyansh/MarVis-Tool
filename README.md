# MarVis Tool
A one-stop open-source tool for all your needs related to Aerial Drone Video Footage to explore various video processing approaches, including video compression, keyframe extraction, object detection, object tracking, and object segmentation. Developed by A. Ancy Micheal, A. R. Revathi, A. Annie Micheal, Divyansh Bhandari, V.R.S.S Karthikeya Jasti.

![Main Page](https://github.com/bh-divyansh/MarVis-Tool/blob/main/assets/MainPage.png?raw=true)

## Installation

We recommend Python version 3.10+ for running this tool, as well as Torch version 2.3+ and torchvision version 0.18+. You can install Python 3.10+ from the [Python website](https://www.python.org/downloads/). Also, please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### Install the tool
Download the tool's code from this github repository or run the following script in your terminal if you have git installed:

    git clone https://github.com/bh-divyansh/MarVis-Tool.git

### Install the dependencies
Navigate to the folder of this repository wherever you have downloaded it by:

    cd MarVis-Tool
Now, Install the dependencies, run this code in your terminal:

    pip install -r requirements.txt

### Download the YOLO Model file
YOLOV8n is recommended. Please download it from the 'Detectors' section of [this repository here](https://github.com/ultralytics/ultralytics).
### Download the YOLOv8-DeepSORT-Object-Tracking folder
Download the YOLOv8-DeepSORT-Object-Tracking folder from [this repository right here](https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking).
#### Go into the cloned folder
    cd YOLOv8-DeepSORT-Object-Tracking
#### Install it's dependencies
    pip install -e '.[dev]'
#### Download the DeepSORT files from the following Google Drive link:

    https://drive.google.com/drive/folders/1kna8eWGrSfzaR6DtNJ8_GchGgPMv3VC8?usp=sharing
After downloading the DeepSORT zip file from the drive, unzip it, go into the subfolders, and place the `deep_sort_pytorch` folder into the `yolo/v8/detect` folder.
### Download SAM weights
Download the `sam_vit_h_4b8939.pth` weights file from [this link](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth).
### Configuring the Paths:
Open the `main.py` file in the MarVis folder and change the following paths according to your computer:

    PATH_TO_YOLO_MODEL = "PATH\\TO\\YOUR\\YOLO_MODEL\\yolov8n.pt"
    DEEPSORT_FOLDER_PATH = "PATH\\TO\\YOUR\\DEEPSORT\\FOLDER"
    ANNOTATIONS_FILE_PATH = "PATH\\TO\\YOUR\\ANNOTATIONS\\FILE.json"
    SAM_HOME_DIR = "PATH\\TO\\YOUR\\SAM\\weights"
### Run the MarVis Tool
Run the below code in your terminal to run the MarVis Tool:

    python main.py


