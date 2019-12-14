import time

import cv2
import numpy as np
import torch

import create_images
import drawing

from detector import Detector
model_path = "detector-1.pth"
model = Detector()
model.load_state_dict(torch.load(model_path))
model.eval()

# Modules imported from this project
from camera.base_camera import BaseCamera

FRAMES_PER_SECOND = 5

class Camera(BaseCamera):

    @staticmethod
    def frames():
        sleep = 1 / FRAMES_PER_SECOND

        while True:
            init_time = time.time()

            canvas = np.ones((400,500,3))

            # Creating image sample that will be predicted using object detector
            sample = create_images.create_sample()
            np_image = sample["image"]
            y_real = np.array(sample["params"])

            X_input = torch.tensor(np_image.transpose(2, 0, 1), dtype=torch.float)

            # Predicted location
            y_pred, c1, c2 = model.predict_with_conv(X_input.unsqueeze(0))

            # Drawing boxes from real and detection
            np_image = drawing.draw_box_image(np_image, y_real[1:], (255, 0, 0))

            # If prediction high enough then the prediction is drawn
            if y_pred[0][0] > 0.8:
                np_image = drawing.draw_box_image(np_image, y_pred[0][1:], (0, 255, 0))

            # Drawing prediction and convolutions
            pred_width, pred_height = np_image.shape[:2]
            canvas[0:pred_height, 0:pred_width, :] = np_image

            # Drawing first convolution
            batch, channels, width, height = c1.shape
            index = 0
            for row in range(2):
                for col in range(3):
                    point1_x = pred_width + (col * width)
                    point2_x = pred_width + ((1+col) * width) 
                    point1_y = row * height
                    point2_y = (1+row) * height

                    canvas[point1_y:point2_y, point1_x:point2_x, 0] = c1[0, index, :, :] 
                    canvas[point1_y:point2_y, point1_x:point2_x, 1] = c1[0, index, :, :] 
                    canvas[point1_y:point2_y, point1_x:point2_x, 2] = c1[0, index, :, :] 
                    index += 1

                    cv2.rectangle(
                        canvas,
                        (point1_x, point1_y),
                        (point2_x, point2_y),
                        (255,255,255),
                        1
                    )

            # Drawing second convolution
            batch, channels, width, height = c2.shape
            index = 0
            for row in range(4):
                for col in range(4):
                    point1_x = pred_width + (col * width)
                    point2_x = pred_width + ((1+col) * width) 
                    point1_y = pred_height + row * height
                    point2_y = pred_height + ((1+row) * height)

                    canvas[point1_y:point2_y, point1_x:point2_x, 0] = c2[0, index, :, :] 
                    canvas[point1_y:point2_y, point1_x:point2_x, 1] = c2[0, index, :, :] 
                    canvas[point1_y:point2_y, point1_x:point2_x, 2] = c2[0, index, :, :] 
                    index += 1

                    cv2.rectangle(
                        canvas,
                        (point1_x, point1_y),
                        (point2_x, point2_y),
                        (255,255,255),
                        1
                    )


            yield cv2.imencode('.jpg', canvas)[1].tobytes()

            # Configured sleep time to try to match the speed at
            # which the video is supposed to be played
            total_time = time.time() - init_time
            real_sleep = sleep - total_time

            time.sleep(real_sleep)
