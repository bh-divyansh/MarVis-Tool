import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import os
import psutil
import subprocess
from Katna.video import Video as katnaVideo
from Katna.writer import KeyFrameDiskWriter
from video_player import Video, VideoControls
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from Katna.image import Image as katnaImage
import cv2
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from video_player import Video, VideoControls
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from dataclasses_json import dataclass_json
from supervision import Detections
import cv2
import json
import warnings
from ultralytics.utils.plotting import Annotator
warnings.filterwarnings("ignore")
LARGEFONT = ("Verdana", 35)

# ----------------------------------------------------------------------------------------------------------

# Note: Remember to replace the Paths with the paths of the models and the video files in your system
PATH_TO_YOLO_MODEL = "PATH\\TO\\YOUR\\YOLO_MODEL\\yolov8n.pt"
DEEPSORT_FOLDER_PATH = "PATH\\TO\\YOUR\\DEEPSORT\\FOLDER"
ANNOTATIONS_FILE_PATH = "PATH\\TO\\YOUR\\ANNOTATIONS\\FILE.json"  # for SAM
SAM_HOME_DIR = "PATH\\TO\\YOUR\\SAM\\weights"  # for SAM

# ----------------------------------------------------------------------------------------------------------


class tkinterApp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        self.geometry(
            f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, ObjectDetectionPage, ObjectTrackingPage, SegmentationPage, VideoCompressionPage, ImageResizingPage, KeyframeExtractionPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="MarVis Tool", font=LARGEFONT)
        label.grid(row=0, column=2, padx=625, pady=70, sticky="ew")

        button1 = ttk.Button(self, text="Object Detection",
                             command=lambda: controller.show_frame(ObjectDetectionPage))
        button1.grid(row=1, column=2, padx=10, pady=10, sticky="we")

        button2 = ttk.Button(self, text="Object Tracking",
                             command=lambda: controller.show_frame(ObjectTrackingPage))
        button2.grid(row=2, column=2, padx=10, pady=10, sticky="we")

        button3 = ttk.Button(self, text="Segmentation",
                             command=lambda: controller.show_frame(SegmentationPage))
        button3.grid(row=3, column=2, padx=10, pady=10, sticky="we")

        button4 = ttk.Button(self, text="Video Compression",
                             command=lambda: controller.show_frame(VideoCompressionPage))
        button4.grid(row=4, column=2, padx=10, pady=10, sticky="we")

        button5 = ttk.Button(self, text="Image Resizing",
                             command=lambda: controller.show_frame(ImageResizingPage))
        button5.grid(row=5, column=2, padx=10, pady=10, sticky="we")

        button6 = ttk.Button(self, text="Keyframe Extraction",
                             command=lambda: controller.show_frame(KeyframeExtractionPage))
        button6.grid(row=6, column=2, padx=10, pady=10, sticky="we")


class ObjectDetectionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label_title = tk.Label(
            self, text="Object Detection with YoloV8 (given frame)")
        label_title.pack(pady=10, padx=10)

        btn_select_video = tk.Button(
            self, text="Select Video", command=self.select_video
        )
        btn_select_video.pack(pady=5)

        btn_select_folder = tk.Button(
            self, text="Select Destination Folder", command=self.select_destination_folder
        )
        btn_select_folder.pack(pady=5)

        self.video_player = Video(self)
        self.video_controls = VideoControls(self, self.video_player)
        self.video_player.ui.pack()
        self.video_controls.ui.pack()

        label_scroll = tk.Label(
            self, text="Scroll to the frame where you want to detect objects using YOLO"
        )
        label_scroll.pack(pady=2)

        btn_detect_object = tk.Button(
            self, text="Detect objects in the given frame", command=self.detect_object
        )
        btn_detect_object.pack(pady=5)

        label_iterations = tk.Label(self, text="Training Iterations:")
        label_iterations.pack(pady=2)

        self.iterations_var = tk.IntVar()
        self.iterations_entry = tk.Entry(
            self, textvariable=self.iterations_var)
        self.iterations_entry.pack()

        btn_train = tk.Button(self, text="Train",
                              command=self.trainObjDetection)
        btn_train.pack(pady=2, padx=5)

        button1 = ttk.Button(self, text="Home Page",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=4)
        self.video_path = ""
        self.dest_folder_path = ""

    def trainObjDetection(self):
        print(
            f"Training with {self.iterations_var.get()} iterations (implement this soon)")

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File", filetypes=(("Video files", "*.mp4 *.avi"), ("All files", "*.*"))
        )
        if file_path:
            self.video_path = file_path
            self.video_player.set_video(self.video_path)

    def select_destination_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Destination Folder")
        if folder_path:
            self.dest_folder_path = folder_path

    def detect_object(self):
        if self.video_path and self.dest_folder_path:
            model = YOLO(PATH_TO_YOLO_MODEL)
            video_path = self.video_path
            frame_number = self.video_controls.get_current_frame()
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Error: Couldn't open the video file")
                exit()

            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read the first frame")
                exit()

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read the frame")
                exit()

            results = model.predict(frame)
            for r in results:
                annotator = Annotator(frame)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])
            img = annotator.result()

            frame_path = os.path.join(
                self.dest_folder_path, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, img)

            print("The objects have been detected.")
            messagebox.showinfo(
                "Object Detection", "Objects detected successfully!")


class ObjectTrackingPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        object_trackers = tk.Frame(self)
        tk.Label(object_trackers, text='Detection and Tracking',
                 font=(0, 18)).pack()
        self.status_label = tk.Label(object_trackers, text="Ready.")
        self.status_label.pack()

        label_title = tk.Label(
            self, text="Object Detection and Tracking using YOLOv8n and DeepSORT")
        label_title.pack(pady=10, padx=10)

        btn_select_video = tk.Button(
            self, text="Select Video", command=self.select_video)
        btn_select_video.pack(pady=5)

        self.video_player = Video(self)
        self.video_controls = VideoControls(self, self.video_player)
        self.video_player.ui.pack()
        self.video_controls.ui.pack()

        btn_tracking_video = tk.Button(
            self, text="Start Tracking", command=self.start_detection_tracking
        )
        btn_tracking_video.pack(pady=5)

        btn_tracking_video_stop = tk.Button(
            self, text="Stop Tracking", command=self.stop_detection_tracking
        )
        btn_tracking_video_stop.pack(pady=5)

        status_label = tk.Label(self, text="Ready.")
        status_label.pack()

        button1 = ttk.Button(self, text="Home Page",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=10)
        self.video_path = ""
        self.video_frames = []

    def stop_detection_tracking(self):
        for proc in psutil.process_iter():
            try:
                if proc.name() == 'python.exe' and 'predict.py' in proc.cmdline():
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    def start_detection_tracking(self):
        self.status_label.config(
            text="Starting Detection and Tracking, please wait...")
        video_path = self.video_path
        subprocess.Popen(
            ['cmd', '/c', f'cd {DEEPSORT_FOLDER_PATH}\\ultralytics\\yolo\\v8\\detect && python predict.py model=yolov8n.pt source=' + video_path + ' show=True'])
        self.status_label.config(
            text="Detection and Tracking stopped. Ready to start again.")

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=(
            ("Video files", "*.mp4 *.avi"), ("All files", "*.*")))
        if file_path:
            self.video_path = file_path
            self.video_player.set_video(self.video_path)


class SegmentationPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label_title = tk.Label(self, text="Video Segmentation")
        label_title.pack(pady=10, padx=10)

        btn_select_video = tk.Button(
            self, text="Select Video", command=self.select_video
        )
        btn_select_video.pack(pady=5)

        btn_select_folder = tk.Button(
            self, text="Select Destination Folder", command=self.select_destination_folder
        )
        btn_select_folder.pack(pady=5)

        self.video_player = Video(self)
        self.video_controls = VideoControls(self, self.video_player)
        self.video_player.ui.pack()
        self.video_controls.ui.pack()

        btn_segment_video = tk.Button(
            self, text="Segment Video", command=self.segment_video
        )
        btn_segment_video.pack(pady=5)

        label_iterations = tk.Label(self, text="Training Iterations:")
        label_iterations.pack(pady=5)

        self.iterations_var = tk.IntVar()
        self.iterations_entry = tk.Entry(
            self, textvariable=self.iterations_var)
        self.iterations_entry.pack()

        btn_train = tk.Button(self, text="Train",
                              command=self.trainSegmentation)
        btn_train.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Home Page",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=10)
        self.video_path = ""
        self.dest_folder_path = ""

    def trainSegmentation(self):
        print(
            f"Training with {self.iterations_var.get()} iterations (implement this soon)")

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File", filetypes=(("Video files", "*.mp4 *.avi"), ("All files", "*.*"))
        )
        if file_path:
            self.video_path = file_path
            self.video_player.set_video(self.video_path)

    def select_destination_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Destination Folder")
        if folder_path:
            self.dest_folder_path = folder_path

    def segment_video(self):
        if self.video_path and self.dest_folder_path:
            model = YOLO(
                PATH_TO_YOLO_MODEL)
            video_path = self.video_path
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Error: Couldn't open the video file")
                exit()

            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read the first frame")
                exit()
            height, width, layers = frame.shape
            size = (width, height)

            fc = 0
            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                results = model.predict(source=frame, conf=0.25)[0]
                ground_truth = sv.Detections.from_ultralytics(results)

                ground_truth.class_id = ground_truth.class_id - 1

                image_bgr = frame
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                bounding_box_annotator = sv.BoundingBoxAnnotator(
                    color=sv.Color.red(), color_lookup=sv.ColorLookup.INDEX)
                mask_annotator = sv.MaskAnnotator(
                    color=sv.Color.red(), color_lookup=sv.ColorLookup.INDEX)

                annotated_frame_ground_truth = bounding_box_annotator.annotate(
                    scene=image_bgr.copy(), detections=ground_truth)

                mask_predictor = SamPredictor(sam)

                mask_predictor.set_image(image_rgb)
                try:
                    masks, scores, logits = mask_predictor.predict(
                        box=ground_truth.xyxy[0],
                        multimask_output=True
                    )

                    detections = sv.Detections(
                        xyxy=sv.mask_to_xyxy(masks=masks),
                        mask=masks
                    )
                    detections = detections[detections.area == np.max(
                        detections.area)]

                    annotated_image = mask_annotator.annotate(
                        scene=image_bgr.copy(), detections=detections)
                except IndexError:
                    annotated_image = frame
                if not os.path.exists(f'{self.dest_folder_path}//temp//'):
                    os.makedirs(f'{self.dest_folder_path}//temp//')

                img_path = f'{self.dest_folder_path}//temp//frame{fc}.jpg'
                cv2.imwrite(img_path, annotated_image)
                print(f"Saved frame {fc} to {img_path}")
                fc += 1
            cap.release()
            filenames = glob.glob(f'{self.dest_folder_path}//temp//*.jpg')
            print(
                f"Found {len(filenames)} files in {self.dest_folder_path}//temp")
            filenames.sort(key=extract_frame_number)

            img_array = []

            for filename in filenames:
                img = cv2.imread(filename)
                img_array.append(img)

            out = cv2.VideoWriter(
                f'{self.dest_folder_path}//SAMOutputVideo_Segmented.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

            for i in range(len(img_array)):
                out.write(img_array[i])

            out.release()
            print("The Image array has: "+str(len(img_array))+" frames.")
            print("The video has been segmented.")
            messagebox.showinfo(
                "Video Segmentation", "Video segmented successfully!")


def extract_frame_number(filename):
    return int(filename.split('frame')[-1].split('.jpg')[0])


@dataclass_json
@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: str


@dataclass_json
@dataclass
class COCOImage:
    id: int
    width: int
    height: int
    file_name: str
    license: int
    date_captured: str
    coco_url: Optional[str] = None
    flickr_url: Optional[str] = None


@dataclass_json
@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: Tuple[float, float, float, float]
    iscrowd: int


@dataclass_json
@dataclass
class COCOLicense:
    id: int
    name: str
    url: str


@dataclass_json
@dataclass
class COCOJson:
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]
    licenses: List[COCOLicense]


def load_coco_json(json_file: str) -> COCOJson:

    with open(json_file, "r") as f:
        json_data = json.load(f)

    return COCOJson.from_dict(json_data)


class COCOJsonUtility:
    @staticmethod
    def get_annotations_by_image_id(coco_data: COCOJson, image_id: int) -> List[COCOAnnotation]:
        return [annotation for annotation in coco_data.annotations if annotation.image_id == image_id]

    @staticmethod
    def get_annotations_by_image_path(coco_data: COCOJson, image_path: str) -> Optional[List[COCOAnnotation]]:
        image = COCOJsonUtility.get_image_by_path(coco_data, image_path)
        if image:
            return COCOJsonUtility.get_annotations_by_image_id(coco_data, image.id)
        else:
            return None

    @staticmethod
    def get_image_by_path(coco_data: COCOJson, image_path: str) -> Optional[COCOImage]:
        for image in coco_data.images:
            if image.file_name == image_path:
                return image
        return None

    @staticmethod
    def annotations2detections(annotations: List[COCOAnnotation]) -> Detections:
        class_id, xyxy = [], []

        for annotation in annotations:
            x_min, y_min, width, height = annotation.bbox
            class_id.append(annotation.category_id)
            xyxy.append([
                x_min,
                y_min,
                x_min + width,
                y_min + height
            ])

        return Detections(
            xyxy=np.array(xyxy, dtype=int),
            class_id=np.array(class_id, dtype=int)
        )


CHECKPOINT_PATH = os.path.join(SAM_HOME_DIR, "sam_vit_h_4b8939.pth")
# should print True if the path is correct and SAM Weights are present
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))
torch.cuda.set_device(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](
    checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)


class VideoCompressionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label_title = tk.Label(self, text="Video Compression")
        label_title.pack(pady=10, padx=10)

        crf_info_label = tk.Label(
            self,
            text="""Enter the Constant Rate Factor value:
            0 means lossless quality and 51 is the worst quality, we recommend you to keep it around 30""",
            justify=tk.CENTER,
        )
        crf_info_label.pack(pady=5)

        self.crf_value_entry = tk.Entry(self)
        self.crf_value_entry.pack(pady=5)

        btn_select_video = tk.Button(
            self, text="Select Video", command=self.select_video
        )
        btn_select_video.pack(pady=5)

        btn_select_folder = tk.Button(
            self, text="Select Destination Folder", command=self.select_destination_folder
        )
        btn_select_folder.pack(pady=5)

        self.video_player = Video(self)
        self.video_controls = VideoControls(self, self.video_player)
        self.video_player.ui.pack()
        self.video_controls.ui.pack()

        btn_compress_video = tk.Button(
            self, text="Compress Video", command=self.compress_video
        )
        btn_compress_video.pack(pady=5)

        self.video_path = ""
        self.dest_folder_path = ""
        button1 = ttk.Button(self, text="Home Page",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=10)

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File", filetypes=(("Video files", "*.mp4 *.avi"), ("All files", "*.*"))
        )
        if file_path:
            self.video_path = file_path
            self.video_player.set_video(self.video_path)

    def select_destination_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Destination Folder")
        if folder_path:
            self.dest_folder_path = folder_path

    def compress_video(self):
        if self.video_path and self.dest_folder_path:
            vd = katnaVideo()
            try:
                crf_value = self.crf_value_entry.get()
                try:
                    crf_value = int(crf_value)
                    if crf_value not in range(0, 52):
                        raise ValueError("CRF value must be between 0 and 51")
                except ValueError:
                    messagebox.showerror(
                        "Error", "Invalid CRF value. Please enter a number between 0 and 51.")
                    return

                output_folder_video_image = self.dest_folder_path
                out_dir_path = os.path.join(".", output_folder_video_image)

                if not os.path.isdir(out_dir_path):
                    os.mkdir(out_dir_path)

                video_file_path = self.video_path

                status = vd.compress_video(
                    crf_parameter=crf_value,
                    file_path=video_file_path,
                    out_dir_path=out_dir_path, force_overwrite=True
                )
                messagebox.showinfo(
                    "Video Compression",
                    f"Video compressed successfully! Saved at: {out_dir_path}",
                )
            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred during video compression: {str(e)}"
                )
        else:
            messagebox.showerror(
                "Error", "Please choose a video file and destination folder."
            )


class ImageResizingPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label_title = tk.Label(self, text="Image Resizing (PNG Images)")
        label_title.pack(pady=10, padx=10)

        Image_Width_Label = tk.Label(
            self,
            text="Resized Image Width",
            justify=tk.LEFT,
        )
        Image_Width_Label.pack(pady=5)

        self.Image_Width = tk.Entry(self)
        self.Image_Width.pack(pady=5)

        Image_Height_Label = tk.Label(
            self,
            text="Resized Image Height",
            justify=tk.LEFT,
        )
        Image_Height_Label.pack(pady=5)

        self.Image_Height = tk.Entry(self)
        self.Image_Height.pack(pady=5)

        Down_Sample_Label = tk.Label(
            self,
            text="Down Sample Factor",
            justify=tk.LEFT,
        )
        Down_Sample_Label.pack(pady=5)

        self.Down_Sample_Factor = tk.Entry(self)
        self.Down_Sample_Factor.pack(pady=5)

        btn_select_image = tk.Button(
            self, text="Select image", command=self.select_image
        )
        btn_select_image.pack(pady=5)

        btn_select_folder = tk.Button(
            self, text="Select Destination Folder", command=self.select_destination_folder
        )
        btn_select_folder.pack(pady=5)

        self.selected_image_label = tk.Label(self, text="Selected Image:")
        self.selected_image_label.pack(pady=10)

        self.image_displayer = ImageDisplayer(self)
        self.image_displayer.pack(pady=3)

        btn_resize_image = tk.Button(
            self, text="Resize the image", command=self.resize_image
        )
        btn_resize_image.pack(pady=5)

        self.image_path = ""
        self.dest_folder_path = ""

        button1 = ttk.Button(self, text="Home Page",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File", filetypes=(("Image files", "*.png"), ("All files", "*.*"))
        )
        if file_path:
            self.image_path = file_path
            self.image_displayer.set_image(self.image_path)

    def select_destination_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Destination Folder")
        if folder_path:
            self.dest_folder_path = folder_path

    def resize_image(self):
        if self.image_path and self.dest_folder_path:
            img_module = katnaImage()
            try:
                image_width_value = self.Image_Width.get()
                image_height_value = self.Image_Height.get()
                Down_Sample_Factor = self.Down_Sample_Factor.get()
                try:
                    image_width_value = int(image_width_value)
                    image_height_value = int(image_height_value)
                    Down_Sample_Factor = int(Down_Sample_Factor)
                    if image_width_value < 1 or image_height_value < 1 or Down_Sample_Factor < 1:
                        raise ValueError(
                            "Width/Height/Down Sample Factor value invalid.")
                except ValueError:
                    messagebox.showerror(
                        "Error", "Invalid width/height/Down sample factor value.")
                    return

                output_folder_cropped_image = self.dest_folder_path
                if not os.path.isdir(os.path.join(".", output_folder_cropped_image)):
                    os.mkdir(os.path.join(".", output_folder_cropped_image))

                image_file_path = self.image_path
                resized_image = img_module.resize_image(
                    file_path=image_file_path,
                    target_width=image_width_value,
                    target_height=image_height_value,
                    down_sample_factor=Down_Sample_Factor,
                )
                img_module.save_image_to_disk(
                    resized_image,
                    file_path=output_folder_cropped_image,
                    file_name="resized_image",
                    file_ext=".jpeg",
                )
                messagebox.showinfo(
                    "Image Resized",
                    f"Image Resized successfully! Saved at: {self.dest_folder_path}",
                )
            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred during image resizing: {str(e)}"
                )
        else:
            messagebox.showerror(
                "Error", "Please choose an image file and destination folder."
            )


class ImageDisplayer(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.image_path = ""
        self.canvas = tk.Canvas(self, width=375, height=300)
        self.canvas.pack()

    def set_image(self, image_path):
        self.image_path = image_path
        self.update_image()

    def update_image(self):
        if self.image_path:
            img = tk.PhotoImage(file=self.image_path)
            self.canvas.create_image(20, 20, anchor=tk.NW, image=img)
            self.canvas.image = img


class KeyframeExtractionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label_title = tk.Label(self, text="Keyframe Extraction")
        label_title.pack(pady=10, padx=10)

        btn_select_video = tk.Button(
            self, text="Select Video", command=self.select_video)
        btn_select_video.pack(pady=5)

        btn_select_folder = tk.Button(
            self, text="Select Destination Folder", command=self.select_destination_folder)
        btn_select_folder.pack(pady=5)

        self.video_player = Video(self)
        self.video_controls = VideoControls(self, self.video_player)
        self.video_player.ui.pack()
        self.video_controls.ui.pack()

        btn_extract_keyframes = tk.Button(
            self, text="Extract Keyframes", command=self.extract_keyframes)
        btn_extract_keyframes.pack(pady=5)

        self.video_path = ""
        self.dest_folder_path = ""
        self.video_frames = []

        button1 = ttk.Button(self, text="Home Page",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=10)

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=(
            ("Video files", "*.mp4 *.avi"), ("All files", "*.*")))
        if file_path:
            self.video_path = file_path
            self.video_player.set_video(self.video_path)

    def select_destination_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Destination Folder")
        if folder_path:
            self.dest_folder_path = folder_path

    def extract_keyframes(self):
        if self.video_path and self.dest_folder_path:

            vd = katnaVideo()

            no_of_frames_to_returned = 12

            diskwriter = KeyFrameDiskWriter(location=self.dest_folder_path)

            try:

                vd.extract_video_keyframes(
                    no_of_frames=no_of_frames_to_returned, file_path=self.video_path, writer=diskwriter
                )
                messagebox.showinfo("Keyframe Extraction",
                                    "Keyframes extracted successfully!")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred during keyframe extraction: {str(e)}")
        else:
            messagebox.showerror(
                "Error", "Please choose a video file and destination folder.")


app = tkinterApp()
app.mainloop()
