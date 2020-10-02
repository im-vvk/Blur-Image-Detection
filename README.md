# Blur-Image-Detection

Classification of Blurred and Non-Blurred Images  
Test Accuracy: **87.57%**

**CERTH Image Blur Dataset**

> E. Mavridaki, V. Mezaris, "No-Reference blur assessment in natural images using Fourier transform and spatial pyramids", Proc. IEEE International Conference on Image Processing (ICIP 2014), Paris, France, October 2014.

The dataset consists of undistorted, naturally-blurred and artificially-blurred images for image quality
assessment purposes.

## How To Use This

1. Download the dataset from here:
   http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip
2. Unzip and load the files into a directory named **CERTH_ImageBlurDataset**
3. Run `pip install -r requirements.txt` to install dependencies
4. Run `python train_script.py` to train the Model
5. Run `python test_script.py` to evaluate the Model on Evaluation Set
6. Run `python -W ignore predict_an_img.py [Image Path]` to check Individual Image
7. If you have run the above commands Train Features, Test Features would be saved in `data` folder and model in `model` folder. You can use it further for any reference and evaluate without again calculating all the features of Train and Test Data and could save some time.

# How This Works

## The basic approach is this

1. use Laplace filter to find edges in the input image
2. compute the variance and the maximum over the pixel values of the filtered image.
3. high variance (and a high maximum) suggest clearly distinguished edges, i.e. a sharp image. Low variance suggests a blurred image
4. use a classifier (SVM) to discriminate the features of a Blurred and non-Blurred

### **Why Laplacian**

**Laplace Filter:** It is an edge detection operator based on gradient methods. it calculates the second derivative of the data. It internally calls the sobel operator for first derivative.

**Sobel operator:** It is also an edge detection operator based on gradient method. i.e., the first order derivative method. Along x or along y or bi-directional. The edges are detected by convolving the kernel with actual image.

**_Note:_** Sobel filter is not used here. Because, By performing EDA it is founded that Laplace filter are (much) better in discriminating between the sharp and blurry images in our data set.

### **Why SVM**

A support vector machine is an algorithm that computes a **best** separating line for us. The line is optimal in the sense that the margin between the two classes along the line is maximal.

### How the code works

1. `python train_script.py` first of all extract the features [laplacian variance and maximum] and then trains the **SVM** model and save in the model folder.
2. `python test_script.py` evaluates the Model on Evaluation Set by the model saved in the **model** folder

**_Note:_** A more compact and illustrative **Jupyter Notebook** is attached named `main.ipynb` for further reference.

---
