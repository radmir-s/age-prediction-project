import face_recognition as fr
import os
import numpy as np


def predict_LR(path):
    image = fr.load_image_file(path)
    face_encodings = np.array(fr.face_encodings(image)[0])
    coeff = np.load("StupidLRcoef.npy")
    intercept = np.load("StupidLRintercept.npy")
    return int(intercept + np.dot(face_encodings, coeff))


if __name__ == "__main__":
    for photo in os.listdir("examples"):
        print(photo, predict_LR(f'examples/{photo}'))
