import face_recognition as fr
import os
import numpy as np

dataset = 'test'

txt = open("test-defected.txt", 'w')

for i in range(1, 101):
    defected = []
    age = f'{i:03}'
    for file in os.listdir('/'.join((dataset, age))):

        image = fr.load_image_file('/'.join((dataset, age, file)))
        face_encodings = fr.face_encodings(image)
        if len(face_encodings) == 1:
            path = '/'.join(('extracted_encodings', dataset, age, file[:-4]))
            np.save(path, np.array(face_encodings[0]))
        else:
            defected.append(file)

    txt.write(f"{len(defected)} defected files in {i:03}: {','.join(defected)}\n")

txt.close()
