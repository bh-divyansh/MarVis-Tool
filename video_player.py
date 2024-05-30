from time import time, sleep

import tkinter as tk
import cv2
from PIL import Image, ImageTk


def ocv2_image_to_tk(image):
    # rearrange colors
    b, g, r = cv2.split(image)
    rearranged_image = cv2.merge((r, g, b))

    # convert into tk format
    pil_image = Image.fromarray(rearranged_image)
    tk_image = ImageTk.PhotoImage(image=pil_image)

    return tk_image


class Video:
    def __init__(self, window: tk.Tk, default_video_path: str = None):
        # set video details
        self.window = window
        self.video_path = None
        self.video = None
        self.current_frame_no = 0
        self.current_frame = None
        self.frame_change_handlers = []
        self.video_change_handlers = []

        # setting up the ui
        self.ui = tk.Frame(self.window)
        self.canvas = tk.Canvas(self.ui)
        self.h_scrollbar = tk.Scrollbar(self.ui, orient='horizontal')
        self.v_scrollbar = tk.Scrollbar(self.ui, orient='vertical')

        self.canvas.config(xscrollcommand=self.h_scrollbar.set,
                           yscrollcommand=self.v_scrollbar.set)
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)

        if default_video_path is not None:
            self.set_video(default_video_path)

    def _set_ui(self):
        if self.video is not None:
            self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas.pack()
        else:
            self.v_scrollbar.pack_forget()
            self.h_scrollbar.pack_forget()
            self.canvas.pack_forget()

    def _set_canvas(self):
        self.canvas.configure(height=337, width=640, scrollregion=(
            0, 0, self.video_width, self.video_height))
        self.canvas_image = self.canvas.create_image(0, 0, anchor='nw')

    def get_video_path(self):
        return self.video_path

    def set_video(self, video_path: str, fps: int = 30):
        self.video_path = video_path
        if video_path is None or video_path == '':
            self.video = None
        else:
            self.video = cv2.VideoCapture(self.video_path)
            self.video.set(cv2.CAP_PROP_FPS, fps)
            self.current_frame_no = 0
            self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._set_canvas()

        for handler in self.video_change_handlers:
            handler()

        self._set_frame()
        self._set_ui()

    def _set_frame(self):
        if self.video is None:
            return
        if self.current_frame_no < 0 or self.current_frame_no >= self.num_frames:
            return
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_no)
        self.current_frame = self.video.read()[1]
        frame = ocv2_image_to_tk(self.current_frame)
        self.canvas.image = frame
        self.canvas.itemconfig(self.canvas_image, image=frame)

        for handler in self.frame_change_handlers:
            handler()

    def add_frame_change_handler(self, callback):
        self.frame_change_handlers.append(callback)

    def add_video_change_handler(self, callback):
        self.video_change_handlers.append(callback)

    def set_current_frame(self, frame_no: int):
        self.current_frame_no = frame_no
        self._set_frame()

    def increment_frame(self):
        if self.current_frame_no < self.num_frames:
            self.current_frame_no += 1
            self._set_frame()

    def decrement_frame(self):
        if self.current_frame_no > 0:
            self.current_frame_no -= 1
            self._set_frame()

    def get_current_frame(self):
        return self.current_frame


class VideoControls:
    def __init__(self, window: tk.Tk, video: Video):
        self.window = window
        self.video = video
        # self.is_playing = False
        # setting up the ui
        self.ui = tk.Frame(self.window)
        self.next_button = tk.Button(
            self.ui, text="Next Frame", command=self.next_frame)
        self.prev_button = tk.Button(
            self.ui, text="Previous Frame", command=self.prev_frame)
        self.frame_slider = tk.Scale(
            self.ui, from_=0, orient='horizontal', command=self.set_frame_from_slider)
        self.frame_input_value = tk.StringVar()
        self.frame_input_value.trace("w", self.manage_frame_input_confirm)
        self.frame_input = tk.Entry(
            self.ui, textvariable=self.frame_input_value)
        self.frame_input.bind('<Return>', self.set_frame_from_entry)
        self.frame_input_confirm = tk.Button(
            self.ui, text="Set Frame", command=self.set_frame_from_entry)
        # self.play_pause_button = tk.Button(self.ui, text='Play')

        self.video.add_video_change_handler(self.update_slider_for_video)
        self.video.add_video_change_handler(self._set_ui)

    def _set_ui(self):
        if self.video.video is not None:
            self.prev_button.pack(side=tk.LEFT)
            self.next_button.pack(side=tk.RIGHT)
            self.frame_slider.pack()
            self.frame_input.pack()
            self.frame_input_confirm.pack()
            # self.play_pause_button.pack()
            # self.play_pause_button.config(text='Pause' if self.is_playing else 'Play')
            self.update_frame_controls()
        else:
            self.prev_button.pack_forget()
            self.next_button.pack_forget()
            self.frame_slider.pack_forget()
            self.frame_input.pack_forget()
            self.frame_input_confirm.pack_forget()

    def get_current_frame(self):
        return self.video.current_frame_no

    def update_slider_for_video(self):
        self.frame_slider.config(
            to=self.video.num_frames, length=self.video.video_width/2)

    def update_frame_controls(self):
        if self.video.current_frame_no == 0:
            self.prev_button['state'] = "disabled"
        else:
            self.prev_button['state'] = "active"
        if self.video.current_frame_no == self.video.num_frames:
            self.next_button['state'] = "disabled"
        else:
            self.next_button['state'] = "active"

        self.frame_slider.set(self.video.current_frame_no)
        self.frame_input.delete(0, tk.END)
        self.frame_input.insert(0, self.video.current_frame_no)

    def set_frame_from_slider(self, event):
        self.video.set_current_frame(int(self.frame_slider.get()))
        self.update_frame_controls()

    def set_frame_from_entry(self, *args):
        self.video.set_current_frame(int(self.frame_input_value.get()))
        self.update_frame_controls()

    def manage_frame_input_confirm(self, *args):
        if self.frame_input_value.get().isdigit() and int(self.frame_input_value.get()) != self.video.current_frame_no:
            self.frame_input_confirm['state'] = 'active'
        else:
            self.frame_input_confirm['state'] = 'disabled'

    def next_frame(self):
        self.video.increment_frame()
        self.update_frame_controls()

    def prev_frame(self):
        self.video.decrement_frame()
        self.update_frame_controls()

    # def toggle_play(self):
    #     self.is_playing = not self.is_playing
    #     if self.is_playing:
    #         self.play()

    # def play(self):
    #     t = time()
    #     fps = 30
    #     frame_time_difference = 1/30
    #     while self.video.current_frame_no < self.video.num_frames:
    #         sleep(frame_time_difference - (time() - t))
    #         t = time()
    #         self.next_frame()
