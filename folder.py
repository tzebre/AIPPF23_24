import os
import cv2
import shutil
import json
import numpy as np
import argparse
def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Haar face detection on image')
    parser.add_argument('--blur', '-b', type=int, default=0,
                        help='add radom pixel to image')
    parser.add_argument('--rotate', '-r', type=int, default=0,
                        help='degrée or rotation')
    return parser

face_path = "dataset/face"
other_path = "dataset/other"
result_path = "dataset/result"
dump = "."
shutil.rmtree(result_path)
    # Recreate the empty directory
os.makedirs(result_path)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
FP = []
FN = []
DP = []
def detect_bounding_box(img):
    faces = face_classifier.detectMultiScale(img, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return faces

def rotate(image, rot):
    height, width = image.shape[:2]
    centerX, centerY = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D((centerX, centerY), rot, 0.5)
    rotated = cv2.warpAffine(image, M, (width, height))
    return rotated

def bluring(image, blur):
    noise = np.random.normal(0, blur, image.shape)
    noisy_image = image+noise
    image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return image

def main(blur, rot):
    for i,img in enumerate(os.listdir(face_path)):
        if img != ".DS_Store":
            im_array = cv2.imread(f"{face_path}/{img}")
            if blur != 0:
                im_array = bluring(im_array, blur)
            if rot != 0:
                im_array = rotate(im_array, rot)
            faces = detect_bounding_box(
                im_array
            )  # apply the function we created to the video frame
            if len(faces) == 0:
                FN.append(f"{result_path}/face_{i}.jpg")
                cv2.imwrite(f"{result_path}/face_{i}.jpg", im_array)
            elif len(faces) > 1:
                DP.append(f"{result_path}/face_{i}.jpg")
                cv2.imwrite(f"{result_path}/face_{i}.jpg", im_array)
            else:
                if i < 5:
                    cv2.imwrite(f"{dump}/face_{i}.jpg", im_array)
    for i,img in enumerate(os.listdir(other_path)):
        if img != ".DS_Store":
            im_array = cv2.imread(f"{other_path}/{img}")
            if blur != 0:
                im_array = bluring(im_array, blur)
            if rot != 0:
                im_array = rotate(im_array, rot)
            faces = detect_bounding_box(
                im_array
            )  # apply the function we created to the video frame
            try:
                if face:
                    FP.append(f"{result_path}/other_{i}.jpg")
                    cv2.imwrite(f"{result_path}/other_{i}.jpg", im_array)
            except:
                pass

    return FN, FP, DP



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    blur = args.blur
    rot = args.rotate
    FN, FP, DP = main(blur, rot)
    res_dict = {"FN":FN, "FP":FP, "DP":DP}
    with open('res.json', 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    for p in res_dict:
        print(p, len(res_dict[p]))
        for f in res_dict[p]:
            """
            im_array = cv2.imread(f)
            cv2.imshow(p,im_array)
            cv2.waitKey(0)
            """
        cv2.destroyAllWindows()
