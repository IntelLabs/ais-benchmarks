import numpy as np
from PIL import Image
import cv2
from matplotlib.animation import FFMpegWriter
from matplotlib import pyplot as plt


class CVideoWriter:
    def __init__(self, filename, fps=30, title="", comment=""):
        self.filename = filename
        self.frames = []
        self.fps = fps

    def add_frame(self, fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame.shape = (h, w, 3)
        self.frames.append(frame)

    def save(self):
        print("Saving video to: %s" % self.filename)
        frame = self.frames[-1]
        w, h, d = frame.shape

        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        video = cv2.VideoWriter(self.filename, fourcc, self.fps, (h, w))

        for i, frame in enumerate(self.frames):
            frame_cv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_cv)
            # Image.frombytes("RGB", (w, h), frame.tobytes()).show()
            # cv2.imshow('frame', frame_cv)
            # cv2.waitKey(100)
            print("frame %d" % i)

        video.release()

    def show(self):
        frame = self.frames[-1]
        w,h,d = frame.shape
        Image.frombytes("RGB", (w, h), frame.tobytes()).show()
