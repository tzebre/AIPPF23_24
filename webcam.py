import cv2
import argparse
import numpy as np
from PIL import Image

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Haar face detection on webcam')
    parser.add_argument('--blur', '-b', type=int, default=0,
                        help='add radom pixel to image')
    return parser

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    blur = args.blur
    while True:
        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully

        if result:
            size = video_frame.shape
            if blur:
                noise = np.random.normal(0, blur, size)
                noisy_image = video_frame+noise
                video_frame = np.clip(noisy_image, 0, 255).astype(np.uint8)

            faces = detect_bounding_box(
                video_frame
            )  # apply the function we created to the video frame
            cv2.imshow(
                    "My Face Detection Project", video_frame
                )  # display the processed frame in a window named "My Face Detection Project"

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows()
