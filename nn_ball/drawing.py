import cv2
import torch
import numpy as np

from detector import Detector
from create_images import create_images
from training import get_batch, draw_box_image

# Module to draw random images using a trained model
# to use:
#   python drawing.py

IMAGES = 100
BATCH_SIZE = 25
test_images = create_images(BATCH_SIZE)

test_data = get_batch(test_images, BATCH_SIZE)
X_test, y_test = next(test_data)

X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)


def draw_random(model, images=10):
    rand_index = np.random.randint(0, X_test_tensor.shape[0], images)
    for i in rand_index:
        # Adding one dimension to the selected image
        X_input = X_test_tensor[i, :, : ,:]
        y_output = y_test_tensor[i, :]

        # Prediction from model
        y_pred = model.predict(X_input.unsqueeze(0))

        # Converting the collected tensor to a numpy array
        # This image is used to draw the circle
        # It has to be transposed to be of shape W x H x Ch
        np_image = X_input.numpy()
        np_image = np_image.transpose(1, 2, 0)

        # Drawing boxes from real and detection
        np_image = draw_box_image(np_image, y_output.numpy()[1:5], (255, 0, 0))

        if y_pred[0][0].item() > 0.8:
            np_image = draw_box_image(np_image, y_pred[0][1:5])

            a=[y_pred[0][5],y_pred[0][6],y_pred[0][7]]
            maxpos = a.index(max(a))

            if maxpos==0:
                advertise = "Blue"
            elif maxpos==1:
                advertise = "Green"
            elif maxpos==2:
                advertise = "Red"
            else:
                advertise = "No color detected"

            np_image = cv2.putText(
                np_image,
                advertise,
                (0, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                1,
                cv2.LINE_AA)

        else:
            np_image = cv2.putText(
                np_image,
                "No detection",
                (0, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                1,
                cv2.LINE_AA)


        cv2.imshow("Pred", np_image)
        cv2.waitKey()


if __name__ == "__main__":
    model_path = "detector.pth"
    model = Detector()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    draw_random(model, IMAGES)
