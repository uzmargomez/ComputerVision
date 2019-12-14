import time
from flask import Flask, Response, jsonify, request

# Modules from this package
from camera.camera import Camera

# Flask module to preview random images 
# Find a way to create a path instead of random images
# Adjust FRAMES_PER_SECOND in base camera to play with 
# the movement of the ball
# Can you calculate the speed of the ball based on the predictions?
#
# to run:
#   python api.py

app = Flask(__name__)

def gen(camera):
    """Video streaming generator function."""
    while True:

        # Getting frame from camera object
        frame = camera.get_frame()

        yield (b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(
        gen(Camera()), 
        mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)