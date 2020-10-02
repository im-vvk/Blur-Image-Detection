from sklearn.metrics import accuracy_score
from sklearn import svm
import os
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tqdm import tqdm

# loading train data


def get_features(path):
    input_size = (512, 512)

    images = os.listdir(path)
    features = []
    for i in tqdm(images):
        feature = []

        # gray = cv2.imread(path+img,0)
        # laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        img = image.load_img(path+i, target_size=input_size)
        # img = image.load_img(path+i)

        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        feature.extend([laplacian.var(), np.amax(laplacian)])

        features.append(feature)
    return pd.DataFrame(features)


path_undis = 'CERTH_ImageBlurDataset/TrainingSet/Undistorted/'
path_art_blur = 'CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred/'
path_nat_blur = 'CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred/'

feature_undis = get_features(path_undis)
print('Undistorted DONE')
feature_art_blur = get_features(path_art_blur)
print('Artificially-Blurred DONE')
feature_nat_blur = get_features(path_nat_blur)
print("Naturally-Blurred DONE")
feature_art_blur.to_csv('./data/art_blur.csv', index=False)
feature_nat_blur.to_csv('./data/nat_blur.csv', index=False)
feature_undis.to_csv('./data/undis.csv', index=False)

# uncomment below code if you have pre-calculated features

# feature_art_blur = pd.read_csv('./data/art_blur.csv')
# feature_nat_blur = pd.read_csv('./data/nat_blur.csv')
# feature_undis = pd.read_csv('./data/undis.csv')

images = pd.DataFrame()

images = pd.DataFrame()
images = images.append(feature_undis)
images = images.append(feature_art_blur)
images = images.append(feature_nat_blur)

x_train = np.array(images)

y_train = np.concatenate((np.zeros((feature_undis.shape[0], )), np.ones(
    (feature_art_blur.shape[0]+feature_nat_blur.shape[0], ))), axis=0)

x_train, y_train = shuffle(x_train, y_train)


# Training model
svm_model = svm.SVC(C=50, kernel='rbf')

svm_model.fit(x_train, y_train)

pred = svm_model.predict(x_train)
print('\nTraining Accuracy:', accuracy_score(y_train, pred))
