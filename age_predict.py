import face_recognition as fr
import numpy as np


def predict_LR(path):
    image = fr.load_image_file(path)
    face_encodings = np.array(fr.face_encodings(image)[0])
    coeff = np.load("LinearRegCoeff.npy")
    intercept = np.load("LinearRegInter.npy")
    return int(intercept + np.dot(face_encodings, coeff))


if __name__ == "__main__":
    print((predict_LR('example.jpg')))
