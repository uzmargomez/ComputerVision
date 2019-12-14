import time
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from detector import Detector
from create_images import create_images

# Module to train a pytorch object
# All the configuration parameters that can be adjusted
# are found in the main function
# to run:
#   python drawing.py

#********************************************************************************************************
# draw_box_image
# Draws the given params from a detection on an image
def draw_box_image(np_image, params, color=(0,0,0)):

    pred_x, pred_y, pred_w, pred_h = params

    point1_x = int(pred_x - pred_w/2)
    point1_y = int(pred_y - pred_h/2)
    point2_x = int(pred_x + pred_w/2)
    point2_y = int(pred_y + pred_h/2)

    np_image = cv2.rectangle(
        np_image,
        (point1_x, point1_y),
        (point2_x, point2_y),
        color,
        1
    )

    return np_image

#*****************************************************************************************************
def get_batch(images, batch_size=5):
    X, y = [], []

    # Creating a list with the data that was collected form the file
    # A portion of this list will be returned by the generator
    for sample in images:
        params = sample["params"]
        canvas = sample["image"]

        # Changing the channels for the network
        # the original image has a dimension of H x W x Ch and pytorch
        # requires the images to be as Ch x H x W
        canvas = canvas.transpose(2, 0, 1)

        y.append(params)
        X.append(canvas)

    # Generator
    # The yield command will return the fraction of the file that was
    # read with the information from the images
    # Each fraction corresponds to the batch size
    for batch in range(0, len(X), batch_size):
        yield X[batch:batch+batch_size], y[batch:batch+batch_size]

#***************************************************************************************************************

def loss_function(y_hat, y_tensor):
    # Evaluation functions for the cost function
    # The Binary Cross entropy is used to calculate the penalty on Pc
    # This is a classification problem

    # The Mean Square Error is calculated for the possition parameters

    # y_hat_position is the correction of the position values
    # when there is an object. It is done using transpose because
    # the whole calculation is done in a batch
    # y_tensor[:, 0] is a tensor of dim [BATCH_SIZE,]
    # y_hat[:, 1:] is a tensor of dim [BATCH_SIZE, 4]
    # The result is a tensor of dim [BATCH_SIZE, 4] but with values
    # equal to zero where there is no object detected
    y_hat_position = (y_tensor[:,0] * y_hat[:, 1:5].T).T
    y_hat_color = (y_tensor[:,0] * y_hat[:, 5:].T).T
    # Prediction that there is an object in the image
    y_hat_pc = y_hat[:, 0]

    label = torch.max(y_tensor[:,5:], 1)[1]
    # Loss function
    # A mixture of binary cross entropy and a MSE where there are
    # detections
    # A large factor is used to adjust the BCE. Without the factor
    # the CE is too small to produce a change in the gradients
    loss = (1000*F.binary_cross_entropy_with_logits(y_hat_pc, y_tensor[:,0]) +
            F.mse_loss(y_hat_position, y_tensor[:,1:5])) + F.cross_entropy(y_hat_color,label)

    return loss


def train_model(model, train_images, test_images, epochs=10, batch_size=100, backup_model=False, draw=False, images=10):
    # Test data from file
    test_data = get_batch(test_images, batch_size)
    X_test, y_test = next(test_data)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    prev_loss = 0
    for epoch in range(epochs):
        init_time = time.time()

        print("Epoch: {}".format(epoch))

        # The generator is used to extract information from the
        # training file that was created with all the generated images
        generator = get_batch(train_images, batch_size)

        losses = []
        for batch, pair in enumerate(generator):
            X, y = pair

            # Converting collected array to tensor
            # X represents the image and y the output
            X_tensor = torch.tensor(X, dtype=torch.float)
            y_tensor = torch.tensor(y, dtype=torch.float)

            # Prediction using the neural network
            y_hat = model(X_tensor)

            # Zeroing gradients
            optimizer.zero_grad()

            # Calculating loss
            loss = loss_function(y_hat, y_tensor)

            # Gradient calculation
            loss.backward()

            # Gradient correction
            optimizer.step()

            # Storing loss to compare at the end of epoch
            losses.append(loss.item())
            print("\tBatch {:03} - loss: {:0.2f}".format(batch, loss.item()))


        # Mean loss in all the batches
        mean_loss = sum(losses) / len(losses)
        print("\tMean loss:{:0.2f}".format(mean_loss))

        if backup_model:
            print("Backing up model")
            torch.save(
                model.state_dict(),
                "backup-{}.pth".format(epoch))

        # Calculating error for test data after epoch
        with torch.no_grad():
            y_test_hat = model(X_test_tensor)
            loss = loss_function(y_test_hat, y_test_tensor)

            total_time = time.time() - init_time
            print("Test loss: {:0.2f}\t Previous Loss: {:0.2f}\t Time: {:0.2f}s".format(loss.item(), prev_loss, total_time))
            prev_loss = loss

            # Drawing some of the predictions after an epoch
            if draw == True:
                rand_index = np.random.randint(0, X_test_tensor.shape[0], images)
                for i in rand_index:
                    # Adding one dimension to the selected image
                    X_pred = X_test_tensor[i, :, : ,:].unsqueeze_(0)
                    y_real = y_test_tensor[i, :].numpy()
                    y_pred = model.predict(X_pred)

                    # Converting and transposing the test image
                    x_img = X_test_tensor[i, :, : ,:].numpy()
                    x_img = x_img.transpose(1, 2, 0)

                    # Creating the bounding box
                    if y_pred[0][0].item() > 0.6:
                        x_img = draw_box_image(x_img, y_pred[0][1:5])
                        if y_pred[0][7].item()>0.6 and y_pred[0][6]<0.6 and y_pred[0][5]<0.6:
                            advertise = "Red"

                        elif y_pred[0][7].item()<0.6 and y_pred[0][6]<0.6 and y_pred[0][5]>0.6:
                            advertise = "Blue"

                        elif y_pred[0][7].item()<0.6 and y_pred[0][6]>0.6 and y_pred[0][5]<0.6:
                            advertise = "Green"
                        else:
                            advertise = "No color detected"

                        x_img = cv2.putText(
                            x_img,
                            advertise,
                            (0, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA)

                    else:
                        x_img = cv2.putText(
                            x_img,
                            "No detection",
                            (0, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA)

                    x_img = draw_box_image(x_img, y_real[1:5], (0,0,0))

                    cv2.imshow("Pred", x_img)
                    cv2.waitKey()


if __name__ == "__main__":

    epochs = 10
    images_samples = 10000
    batch_size = 100
    load_model = True #True
    save_path = "detector.pth"
    model_path = "detector.pth"

    # Pytorch model to be trained
    circle_detector = Detector()
    if load_model:
        print("Loading previusly trained model {}".format(model_path))
        circle_detector.load_state_dict(torch.load(model_path))

    train_images = create_images(images_samples)
    test_images = create_images(batch_size)

    # Training function
    try:
        train_model(
            model = circle_detector,
            train_images=train_images,
            test_images=test_images,
            epochs=epochs,
            batch_size=batch_size,
            backup_model=True,
            draw=False,
            images=5)

        # Saving model coefficients
        print("Saving model")
        torch.save(
            circle_detector.state_dict(),
            save_path)

    except KeyboardInterrupt:

        # Saving model coefficients
        print("Stoping training. Saving the model.")
        torch.save(
            circle_detector.state_dict(),
            save_path)
