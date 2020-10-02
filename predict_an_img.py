import cv2
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import joblib
import sys

if len(sys.argv) != 2:
    msg = 'Error: Please Provide an Image Path in Command line'
    print("\033[91m {}\033[00m" .format(msg))
    exit()

image_path = sys.argv[1]
input_size = (512, 512)
img = image.load_img(image_path, target_size=input_size)

gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
img_features = [[laplacian.var(), np.amax(laplacian)]]

model = joblib.load('./model/model.pkl')

msg = '\nModel Prediction: ' + ('Undistorted' if (
    model.predict(img_features)[0] == 0) else 'Blurred\n')
print("\033[96m {}\033[00m" .format(msg))
# plt.imshow(img)
# plt.title('Prediction: ' +
#           ('Undistorted' if (model.predict(img_features)[0] == 0) else 'Blurred'))
# plt.xticks([])
# plt.yticks([])
# plt.show()
