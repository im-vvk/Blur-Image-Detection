from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib


x_test = []
y_test = []

dgbset = pd.read_excel(
    'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
nbset = pd.read_excel(
    'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')

dgbset['MyDigital Blur'] = dgbset['MyDigital Blur'].apply(lambda x: x.strip())
dgbset = dgbset.rename(index=str, columns={"Unnamed: 1": "Blur Label"})

nbset['Image Name'] = nbset['Image Name'].apply(lambda x: x.strip())

folder_path = 'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'

input_size = (512, 512)
# load image arrays
for file_name in tqdm(os.listdir(folder_path)):
    if file_name != '.DS_Store':

        feature = []

        img = image.load_img(folder_path+file_name, target_size=input_size)
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        feature.extend([laplacian.var(), np.amax(laplacian)])

        x_test.append(feature)
        blur = dgbset[dgbset['MyDigital Blur']
                      == file_name].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(file_name, 'not a pic')

folder_path = 'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'

# load image arrays
for file_name in tqdm(os.listdir(folder_path)):
    if file_name != '.DS_Store':
        feature = []

        img = image.load_img(folder_path+file_name, target_size=input_size)
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        feature.extend([laplacian.var(), np.amax(laplacian)])

        x_test.append(feature)

        blur = nbset[nbset['Image Name'] ==
                     file_name.split('.')[0]].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(file_name, 'not a pic')

test_df = pd.DataFrame(x_test)
test_df['blur_label'] = y_test
test_df.columns = ['laplacian_var', 'laplacian_max', 'blur_label']
test_df.to_csv('./data/test_data.csv', index=False)

# uncomment below code if you have pre-calculated features
# test_df = pd.read_csv('./data/test_data.csv')

model = joblib.load('./model/model.pkl')

pred = model.predict(np.array(test_df[['laplacian_var', 'laplacian_max']]))
print('Testing Accuracy:', accuracy_score(
    np.array(test_df['blur_label']), pred))
