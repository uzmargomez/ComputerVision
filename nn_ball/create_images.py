import random
import cv2
import numpy as np

# Module to create random images
# It is called by training, drawing and the flask api

IMAGE_SIZE = 200

def create_sample():
    canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3)) * 255

    # If there is object
    pc = random.randint(0, 1)

    # Circle position and dimension
    radius = int(random.randint(0, IMAGE_SIZE) / 2)
    radius = max(int(IMAGE_SIZE*0.05), radius)
    radius = min(int(IMAGE_SIZE*0.4), radius)

    center_x = random.randint(0, IMAGE_SIZE)
    center_x = max(center_x, radius)
    center_x = min(center_x, (IMAGE_SIZE-radius))

    center_y = random.randint(0, IMAGE_SIZE)
    center_y = max(center_y, radius)
    center_y = min(center_y, (IMAGE_SIZE-radius))

    colors = [(255,0,0),(0,255,0),(0,0,255)]

    color = random.choice(colors)

    if pc == 1:
        cv2.circle(
            canvas,
            (center_x, center_y),
            radius,
            color,
            -1
        )

    sample = {
        "params": [pc, pc*center_x, pc*center_y, pc*radius*2, pc*radius*2, pc*(color[0]/255),
                    pc*(color[1]/255), pc*(color[2]/255)],
        #"params": [center_x, center_y, radius*2, radius*2],
        "image": canvas,
    }

    return sample


def create_images(sample_size):
    images = []
    for i in range(sample_size):
        sample = create_sample()
        images.append(sample)

    return images
