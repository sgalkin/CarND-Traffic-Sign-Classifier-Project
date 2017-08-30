
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:


# Load pickled data
import pickle
import numpy as np

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))


# In[2]:


import csv

def read_names(filename):
    with open(filename, 'r') as names:
        return dict(((int(a), b) for a, b in csv.reader(names) if a.isdigit()))

names_file = 'signnames.csv'
sign_names = read_names(names_file)


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[3]:


import numpy as np

# Number of training examples
n_train = len(y_train)

# Number of validation examples
n_validation = len(y_valid)

# Number of testing examples.
n_test = len(y_test)

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(np.concatenate((y_train, y_valid, y_test))))

print('Number of training examples =', n_train)
print('Number of validation examples =', n_validation)
print('Number of testing examples =', n_test)
print('Image data shape =', image_shape)
print('Number of classes =', n_classes)


# In[4]:


from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[5]:


import random
import math
import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
get_ipython().magic('matplotlib inline')


# In[6]:


# Create and display grid plot
def plot_grid(array, indecies, cols, image_size=(1,1)):

    def normalize(array, min_value=0.0, max_value=1.0):
        return np.dstack(max_value*(a - np.min(a))/(np.max(a) - np.min(a)) + min_value
                         for a in np.dsplit(array, array.shape[-1]))

    mod = len(indecies) % cols
    rows = int(len(indecies) / cols) + (0 if mod == 0 else 1)
    figsize = tuple(a * b for a, b in zip(image_size, (cols, rows)))

    padding = np.tile(np.zeros(array[0].shape), (1, 0 if mod == 0 else cols - mod, 1))
    images = np.hstack((normalize(array[i].copy())) for i in indecies)
    images = np.hstack((images, padding))
    images = np.vstack(np.hsplit(images, rows))

    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(images.squeeze(), 
               cmap=None if len(images[0].shape) > 2 else 'gray',
               interpolation="nearest")


# In[7]:


def plot_bar(xs, ys, names, xlabel=None, ylabel=None, title=None):
    plt.figure(figsize=(16, 4))
    width = 0.28
    n_series = len(ys)
    n_categories = len(np.unique(np.concatenate(xs)))
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.bar(x + (i - int(n_series/2))*width, y, width)
    
    plt.xticks(np.arange(0, n_categories, 2))
    if title: plt.legend(names)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.grid(True)
    if title: plt.title(title)
    
def plot_histogram(series, names, xlabel=None, ylabel=None, title=None):
    xs, ys = zip(*[(v, 100*c/np.sum(c)) 
       for v, c in (
           np.unique(d, return_counts=True) for d in series
       )])
    plot_bar(xs, ys, names, xlabel, ylabel, title)


# In[8]:


# Labels distribution
plot_histogram((y_train, y_valid, y_test), 
               ('train', 'valid', 'test'), 
               xlabel='Category', ylabel='% in set', title='Signs distribution')


# In[9]:


# Take a copy of data just to simplify pre-process editing
X_train_origin, y_train_origin = np.copy(X_train), np.copy(y_train)
X_valid_origin, y_valid_origin = np.copy(X_valid), np.copy(y_valid)
X_test_origin, y_test_origin = np.copy(X_test), np.copy(y_test)


# In[10]:


plot_grid(X_train, 
          np.array([np.argwhere(y_train==i)[0:4] for i in range(n_classes)]).flatten(), 
          16, image_size=(1, 1))


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[11]:


IMAGE_SIZE=(2.4, 2.4)

def plot_images_in_row(images, title=None, image_size=IMAGE_SIZE, norm=None):
    plt.figure(figsize=(len(images)*image_size[0], image_size[1]))
    if title: plt.suptitle(title)

    for i,img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.squeeze(), 
                   cmap='gray' if len(img.shape) < 3 or img.shape[2] == 1 else None,
                   norm=norm, vmin=0., vmax=255.)

    plt.tight_layout(pad=2)


# In[12]:


# Image convertion functions
def passthrough(image):
    return image

def grayscale(image):
    return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), -1)


# In[13]:


for p in (passthrough, grayscale):
    plot_images_in_row((X_train[0], p(X_train[0])), title=p.__name__)


# In[14]:


# Utilities
def rnd(a, b):
    return (b - a) * random.random() + a

def for_each_depth(method, array):
    return (
        np.expand_dims(method(array), axis=-1) if len(array.shape) < 3 else 
        np.stack((method(np.squeeze(a, axis=-1)) 
                 for a in np.dsplit(array, array.shape[-1])), 
                axis=-1))

def for_each_row(method, array):
    r = np.stack((method(np.squeeze(a, axis=0)) 
                     for a in np.vsplit(array, len(array))), 
                    axis=0)
    if len(r.shape) < 4: r = np.expand_dims(r, axis=-1)
    return r


# In[15]:


# Image transformations
def normalize(array, mean, sigma):
    def normalize_layer(a, mean, sigma):
        if mean is None: mean = np.mean(a)
        if sigma is None: sigma = 1.*(np.max(a) - np.min(a))
        return (a - mean) / sigma
    return for_each_depth(lambda a: normalize_layer(a, mean, sigma), array)

def noise(array, sigma=1.0):
    return array + sigma*np.random.standard_normal(array.shape)

def adjust_contrast(array, factor):
    return for_each_depth(lambda a: a * factor + (1 - factor) * np.mean(a), array)

def adjust_brightness(array, value):
    return for_each_depth(lambda a: a + value * (np.max(a) - np.min(a)), array)

def blur(image, kernel):
    return np.expand_dims(cv2.GaussianBlur(image, kernel, 0), -1)
    
def rotate(image, angle, scale):
    rows, cols = image.shape[0:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    return cv2.warpAffine(image, M, (cols, rows))

def shift(image, dx, dy):
    rows, cols = image.shape[0:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (cols, rows))


# In[16]:


gray = grayscale(X_train[0])
plot_images_in_row((gray, normalize(gray, None, None)), 'normalize', norm=clr.NoNorm())
for p in (
        (lambda i: noise(i, math.sqrt(4)), 'noise'),
        (lambda i: adjust_contrast(i, 0.85), 'contrast low'),
        (lambda i: adjust_contrast(i, 1.15), 'contrast high'),
        (lambda i: adjust_brightness(i, -0.2), 'brightness low'),
        (lambda i: adjust_brightness(i, 0.2), 'brightness high'),
        (lambda i: blur(i, (3, 3)), 'blur'),
        (lambda i: rotate(i, -15, 0.8), 'rotate ccw'),
        (lambda i: rotate(i, 15, 1.2), 'rotate cw'),
        (lambda i: shift(i, -2, 2), 'shift')):
    plot_images_in_row((gray, p[0](gray)), title=p[1], image_size=IMAGE_SIZE)


# In[17]:


# Batch transformations
def transform_images(array, transformation, *args, **kwargs):
    return for_each_row(lambda i: transformation(i, *args, **kwargs), array)

def boost(collection, labels, transformation, size):
    return (
        np.vstack(transform_images(collection, transformation) for _ in range(size)),
        np.hstack(labels for _ in range(size)))  

def equalize(features, labels, transformation, scale=1.0):
    values, count = np.unique(labels, return_counts=True)
    max_count = np.max(count)
    
    labels_copy = np.copy(labels)
    for v in values:
        indecies, = np.where(labels_copy==v)
        has = len(indecies)
        need = int(scale * max_count) - has
        
        if has == 0 or need <= 0:
            continue
        
        transform_indecies = np.random.choice(indecies, need)
        features = np.vstack((features, 
                              np.expand_dims(
                                  np.array(list(transformation(features[i]) 
                                                for i in transform_indecies)), axis=-1)))
        labels = np.hstack((labels, np.full((need,), v)))
    
    _, count = np.unique(labels, return_counts=True)
    assert(int(scale * np.max(count)) == np.min(count))
    assert(len(features) == len(labels))

    return features, labels


# In[18]:


# Transformations

NOISE_SIGMA = 3
CONTRAST_ADJUSTMENT = 0.90
EQUALIZATION_FACTOR = 0.33
BOOST_SIZE = 5

def preprocess_transformation(collection, mean=None, sigma=None):
    for processor in (grayscale, lambda i: normalize(i, mean, sigma)):
        collection = transform_images(collection, processor)
    return collection

def boost_transformation(img, mean=None, sigma=None):
    return (
        normalize(
            shift(
                rotate(
                    noise(
                        adjust_brightness(
                            adjust_contrast(
                                passthrough(img
                                ), rnd(0.85, 1.15)
                            ), rnd(-0.1, 0.1)
                        ), math.sqrt(NOISE_SIGMA)
                    ), rnd(-14, 14), rnd(0.8, 1.2)
                ), rnd(-2, 2), rnd(-2, 2)
            ), mean, sigma
        ))

def equalize_tranformation(img):
    return (
        shift(
            rotate(
                adjust_contrast(
                    img, rnd(0.9, 1.1)
                ), rnd(-3, 3), rnd(0.9, 1.1)
            ), rnd(-1, -1), rnd(1, 1)))


# In[19]:


print("origin:\t", X_train_origin.shape)
X_train, y_train = np.copy(X_train_origin), np.copy(y_train_origin)
print("train:\t", X_train.shape)

X_train = transform_images(X_train, grayscale)
if EQUALIZATION_FACTOR:
    X_train, y_train = equalize(X_train, y_train, equalize_tranformation, EQUALIZATION_FACTOR)

X_boost, y_boost = boost(X_train, y_train, boost_transformation, BOOST_SIZE)
X_train = transform_images(X_train, 
                           lambda i:
                               normalize(
                                   noise(
                                       adjust_contrast(
                                           i, CONTRAST_ADJUSTMENT
                                       ), math.sqrt(NOISE_SIGMA)
                                   ), None, None
                               )
                          )
X_train, y_train = np.vstack((X_train, X_boost)), np.hstack((y_train, y_boost))
assert(len(X_train) == len(y_train))
print ("final:\t", X_train.shape)


# In[20]:


plot_grid(X_train, [random.randint(0, len(X_train)) for _ in range(64)], 8, image_size=(2, 2))


# In[21]:


plot_histogram((y_train,), ('train',), xlabel='Categories', ylabel='% in set', title='Equalized distribution')


# In[22]:


X_valid, y_valid = np.copy(X_valid_origin), np.copy(y_valid_origin)
X_valid = preprocess_transformation(X_valid, None, None)


# In[23]:


X_test, y_test = np.copy(X_test_origin), np.copy(y_test_origin)
X_test = preprocess_transformation(X_test, None, None)


# In[24]:


def store(name, features, labels):
    with open(name, 'wb') as output:
        pickle.dump({'features': features, 'labels': labels}, 
                    output, 
                    pickle.HIGHEST_PROTOCOL)

store("train.processed.p", X_train, y_train)
store("valid.processed.p", X_valid, y_valid)
store("test.processed.p", X_test, y_test)


# In[25]:


n_channels = X_train.shape[-1]


# # Model Architecture

# In[26]:


import tensorflow as tf


# In[27]:


from tensorflow.contrib.layers import flatten            
    
def LeNet(spec, x, conv_keep_prob, dense_keep_prob, verbose=False):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    m = 3

    def convolution(x, W, b, names, paddings):
        assert(len(names) == len(paddings))
        for name, padding in zip(names, paddings):
            x = tf.nn.conv2d(x, W[name], strides=[1, 1, 1, 1], padding=padding) + b[name]
            if verbose: print("conv", x)
            x = tf.nn.relu(x)
            if verbose: print("relu", x)
        return x

    def pre(x):
        weights = {
            'c7x7_1': tf.Variable(tf.truncated_normal([7, 7, n_channels, 2**(m + 0)], mu, sigma)),
            'c1x1_1': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 0), 2**(m - 1)], mu, sigma)),
            'c5x5_1': tf.Variable(tf.truncated_normal([5, 5, 2**(m - 1), 2**(m + 2)], mu, sigma)), 
        }
        biases = {
            'c7x7_1': tf.Variable(tf.zeros(2**(m + 0))),
            'c1x1_1': tf.Variable(tf.zeros(2**(m - 1))),
            'c5x5_1': tf.Variable(tf.zeros(2**(m + 2))),
        }
        
        # Preprocessing 
        x = convolution(x, weights, biases, ('c7x7_1', 'c1x1_1', 'c5x5_1',), ('SAME', 'VALID', 'VALID',))
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        if verbose: print("max-pool", x)
        if verbose: print("pre")
        return x
    
    def branch_a(x):
        weights = {
            'c1x1_1': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 2), 2**(m + 1)], mu, sigma)),
            'c5x5_1': tf.Variable(tf.truncated_normal([5, 5, 2**(m + 1), 2**(m + 4)], mu, sigma)),
            'c1x1_2': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 4), 2**(m + 3)], mu, sigma)),
            'c3x3_2': tf.Variable(tf.truncated_normal([3, 3, 2**(m + 3), 2**(m + 5)], mu, sigma)),
        }
        biases = {
            'c1x1_1': tf.Variable(tf.zeros(2**(m + 1))),
            'c5x5_1': tf.Variable(tf.zeros(2**(m + 4))),
            'c1x1_2': tf.Variable(tf.zeros(2**(m + 3))),
            'c3x3_2': tf.Variable(tf.zeros(2**(m + 5))),
        }

        x = convolution(x, weights, biases, ('c1x1_1', 'c5x5_1'), ('VALID', 'VALID'))
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')    
        if verbose: print("max-pool", x)
        
        x = convolution(x, weights, biases, ('c1x1_2', 'c3x3_2'), ('VALID', 'VALID'))
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')   
        if verbose: print("max-pool", x)
        if verbose: print("branch_a")
        return x
    
    def branch_b(x):
        weights = {
            'c1x1_1': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 2), 2**(m + 1)], mu, sigma)),
            'c3x3_1': tf.Variable(tf.truncated_normal([3, 3, 2**(m + 1), 2**(m + 4)], mu, sigma)),
            'c1x1_2': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 4), 2**(m + 2)], mu, sigma)),
            'c3x3_2': tf.Variable(tf.truncated_normal([3, 3, 2**(m + 2), 2**(m + 4)], mu, sigma)),
            'c1x1_3': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 4), 2**(m + 2)], mu, sigma)),
            'c3x3_3': tf.Variable(tf.truncated_normal([3, 3, 2**(m + 2), 2**(m + 4)], mu, sigma)),

            'c1x1_4': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 4), 2**(m + 3)], mu, sigma)),
            'c5x5_4': tf.Variable(tf.truncated_normal([5, 5, 2**(m + 3), 2**(m + 5)], mu, sigma)),
        }
        biases = {
            'c1x1_1': tf.Variable(tf.zeros(2**(m + 1))),
            'c3x3_1': tf.Variable(tf.zeros(2**(m + 4))),
            'c1x1_2': tf.Variable(tf.zeros(2**(m + 2))),
            'c3x3_2': tf.Variable(tf.zeros(2**(m + 4))),
            'c1x1_3': tf.Variable(tf.zeros(2**(m + 2))),
            'c3x3_3': tf.Variable(tf.zeros(2**(m + 4))),
        
            'c1x1_4': tf.Variable(tf.zeros(2**(m + 3))),
            'c5x5_4': tf.Variable(tf.zeros(2**(m + 5))),            
        }

        x = convolution(x, weights, biases, 
                        ('c1x1_1', 'c3x3_1', 'c1x1_2', 'c3x3_2', 'c1x1_3', 'c3x3_3'), 
                        ('VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID'))
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')    
        if verbose: print("max-pool", x)
        
        x = convolution(x, weights, biases, ('c1x1_4', 'c5x5_4'), ('VALID', 'VALID'))
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
        if verbose: print("max-pool", x)
        if verbose: print("branch_b")
        return x

    def branch_c(x):
        weights = {
            'c1x1_1': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 2), 2**(m + 1)], mu, sigma)),
            'c7x7_1': tf.Variable(tf.truncated_normal([7, 7, 2**(m + 1), 2**(m + 3)], mu, sigma)),
            'c1x1_2': tf.Variable(tf.truncated_normal([1, 1, 2**(m + 3), 2**(m + 2)], mu, sigma)),
            'c3x3_2': tf.Variable(tf.truncated_normal([3, 3, 2**(m + 2), 2**(m + 5)], mu, sigma)),
        }
        biases = {
            'c1x1_1': tf.Variable(tf.zeros(2**(m + 1))),
            'c7x7_1': tf.Variable(tf.zeros(2**(m + 3))),
            'c1x1_2': tf.Variable(tf.zeros(2**(m + 2))),
            'c3x3_2': tf.Variable(tf.zeros(2**(m + 5))),            
        }

        x = convolution(x, weights, biases, ('c1x1_1', 'c7x7_1'), ('VALID', 'VALID'))
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')    
        if verbose: print("max-pool", x)
        
        x = convolution(x, weights, biases, ('c1x1_2', 'c3x3_2'), ('VALID', 'VALID'))
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        if verbose: print("max-pool", x)
        if verbose: print("branch_c", x)
        return x
    
    def total_depth(*args):
        return sum(int(x.get_shape()[-1]) for x in args)
        
    def fc(x, out_shape, add_activation=True):    
        W = tf.Variable(tf.truncated_normal([int(x.get_shape()[-1]), out_shape], mu, sigma))
        b = tf.Variable(tf.zeros(int(out_shape)))
        x = tf.nn.xw_plus_b(x, W, b)
        if verbose: print("fc", x)
        x = tf.nn.tanh(x) if add_activation else x
        if verbose: print("tanh" if add_activation else "no-op", x)
        return x

    def fc_full(x): return fc(x, int(x.get_shape()[-1])>>0)
    def fc_half(x): return fc(x, int(x.get_shape()[-1])>>1)
    def fc_quarter(x): return fc(x, int(x.get_shape()[-1])>>2)
    def out(x): return fc(x, n_classes, add_activation=False)

    branches = { 'a': branch_a, 'b': branch_b, 'c': branch_c, }
    denses = { 'f': fc_full, 'h': fc_half, 'q': fc_quarter }

    if verbose: print('input', x)
 
    x = pre(x)
    x = tf.concat(
        tuple(tf.contrib.layers.flatten(branches[b](tf.identity(x))) 
              for b in spec if b in branches), 
        axis=1)
    x = tf.nn.dropout(x, conv_keep_prob)
    if verbose: print("dropout", x)
    
    for d in spec:
        if d in denses:
            x = denses[d](x)
    x = tf.nn.dropout(x, dense_keep_prob)
    if verbose: print("dropout", x)
    
    return out(x)


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[28]:


# Knobs
EPOCHS = 25
BATCH_SIZE = 1280
RATE = 0.001

conv_keep_probability = 0.25
dense_keep_probability = 0.40

SPEC='abch'


# In[95]:


def loss_operation(model):
    logits, _, _, _ = model

    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    return tf.reduce_mean(cross_entropy), y


# In[29]:


def create_model(spec, verbose=False):
    x = tf.placeholder(tf.float32, (None, 32, 32, n_channels))
    conv_keep_prob = tf.placeholder_with_default(1., (None), name='ConvDropout')
    dense_keep_prob = tf.placeholder_with_default(1., (None), name='DenseDropout')
    
    logits = LeNet(spec, x, conv_keep_prob, dense_keep_prob, verbose=verbose)
    
    return logits, x, conv_keep_prob, dense_keep_prob


# In[30]:


def train_operation(model):
    logits, _, _, _ = model

    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=RATE)
    return optimizer.minimize(loss_operation), y
    
def train(operation, model, X_data, y_data, conv_keep_probability, dense_keep_probability):
    training_operation, y = operation
    _, x, conv_keep_prob, dense_keep_prob = model 
    for offset in range(0, len(X_data), BATCH_SIZE):
        batch_x, batch_y = X_train[offset:offset+BATCH_SIZE], y_train[offset:offset+BATCH_SIZE]
        sess.run(training_operation, 
                 feed_dict={x: batch_x,
                            y: batch_y, 
                            conv_keep_prob: conv_keep_probability,
                            dense_keep_prob: dense_keep_probability})


# In[31]:


def accuracy_operation(model):
    logits, _, _, _ = model
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32)), y

def evaluate(operation, model, X_data, y_data, conv_keep_probability=1.0, dense_keep_probability=1.0):
    accuracy_operation, y = operation
    _, x, conv_keep_prob, dense_keep_prob = model
    
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, 
                            feed_dict={x: batch_x, 
                                       y: batch_y,
                                       conv_keep_prob: conv_keep_probability,
                                       dense_keep_prob: dense_keep_probability})
        total_accuracy += (accuracy*len(batch_x))
    return total_accuracy/num_examples


# In[67]:


import datetime as dt

tf.reset_default_graph()

model = create_model(SPEC, verbose=False)
train_op = train_operation(model)
accuracy_op = accuracy_operation(model)

saver = tf.train.Saver()

training_curve = np.empty((EPOCHS, 2))
dropout = (conv_keep_probability, dense_keep_probability)

# Training
with tf.Session() as sess:        
    sess.run(tf.global_variables_initializer())
    print("Training...", SPEC)
    for i in range(EPOCHS):
        epoch_begin = dt.datetime.now()
        train(train_op, model, *shuffle(X_train, y_train), *dropout)
        epoch_end = dt.datetime.now()

        train_accuracy = evaluate(accuracy_op, model, X_train, y_train, *dropout)
        validation_accuracy = evaluate(accuracy_op, model, X_valid, y_valid)

        print("E{:02} {:.3f}s => "
              "Train Accuracy = {:.3f}; Validation Accuracy = {:.3f}".format(
                  i+1, (epoch_end - epoch_begin).total_seconds(), 
                  train_accuracy, validation_accuracy))
        training_curve[i] = (train_accuracy, validation_accuracy)
            
    test_accuracy = evaluate(accuracy_op, model, X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
        
    model_file = saver.save(sess, './lenet-{}-{:.3f}'.format(SPEC, test_accuracy))
    print("Model saved into", model_file)
    
with open('./curve-{}-{:.3f}.p'.format(SPEC, test_accuracy), 'wb') as curve:
    pickle.dump(training_curve, curve)


# In[32]:


MODEL_FILE = './lenet-abch-0.984'
CURVE_FILE = './curve-abch-0.984.p'


# In[33]:


tf.reset_default_graph()

model = create_model(SPEC)
accuracy_op = accuracy_operation(model)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, MODEL_FILE)
    print("train -", evaluate(accuracy_op, model, X_train, y_train))
    print("valid -", evaluate(accuracy_op, model, X_valid, y_valid))
    print("test -", evaluate(accuracy_op, model, X_test, y_test))


# In[34]:


with open(CURVE_FILE, 'rb') as f:
    curve = pickle.load(f)

plt.figure(figsize=(12, 4))
plt.plot(range(1, len(curve)+1), curve)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(('train', 'valid'))
plt.grid()
plt.xticks(range(1, len(curve)+1))
plt.title('Training curve')


# In[35]:


def topK_operation(model, k):
    logits, _, _, _ = model
    softmax_prediction = tf.nn.softmax(logits)
    return tf.nn.top_k(softmax_prediction, k)

def topK(op, model, X_data):
    _, x, _, _ = model
    
    sess = tf.get_default_session()
    return (np.vstack(x) 
            for x in zip(*(
                sess.run(op, feed_dict={x: X_data[offset:offset+BATCH_SIZE]})
                for offset in range(0, len(X_data), BATCH_SIZE))))

tf.reset_default_graph()
model = create_model(SPEC)
top5_op = topK_operation(model, 5)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, MODEL_FILE)
    p_test, pi_test = topK(top5_op, model, X_test)


# In[36]:


def plot_probabilities(image, image_idx, probabilities, probabilities_idx, names):
    plt.figure(figsize=(10, IMAGE_SIZE[0]))
    
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('{}, {}'.format(image_idx, names[image_idx]))
    plt.subplots_adjust(wspace=0, hspace=0.1)
    
    plt.subplot(122)
    ticks = range(len(probabilities))
    plt.bar(ticks, probabilities)
    plt.xticks(ticks, ('{}\n{}'.format(i, names[i].replace(' ', '\n')) for i in probabilities_idx))
    plt.title('probabilities')
    plt.subplots_adjust(wspace=0, hspace=0)


# In[37]:


# 3 most confusing (lowest high probability)
num_to_display = 3
for i in p_test[:,0].argsort()[:num_to_display]:
    plot_probabilities(X_test[i].squeeze(), y_test[i], p_test[i], pi_test[i], sign_names)


# In[38]:


misclassified = np.where(y_test != pi_test[:,0])
y_m = y_test[misclassified]
X_m = X_test[misclassified]
p_m = p_test[misclassified]
pi_m = pi_test[misclassified]


# In[87]:


v, c = np.unique(y_m, return_counts=True)

plt.figure(figsize=(16, 4))
plt.xticks(range(n_classes))
plt.bar(v, c/np.sum(c), label='Misclassified')
plt.bar(range(n_classes), -np.unique(y_test, return_counts=True)[1]/len(y_test), label='Distribution')
plt.xlabel('Category')
plt.ylabel('Fraction')
plt.title('Misclassified distribution')
plt.grid(axis='y')
plt.legend()


# In[93]:


num_to_display = 3
worst_n = 5
for i in v[c.argsort()[-worst_n:]]:
    for j in shuffle(np.where(y_m == i)[0])[:num_to_display]:
        plot_probabilities(X_m[j].squeeze(), y_m[j], p_m[j], pi_m[j], sign_names)


# In[94]:


plot_grid(X_m[y_m.argsort()], range(len(X_m)), 16, image_size=(1, 1))


# In[50]:


from sklearn import metrics

precision = metrics.precision_score(y_test, pi_test[:,0], average=None)
recall = metrics.recall_score(y_test, pi_test[:,0], average=None)


# In[68]:


plt.figure(figsize=(16, 4))
plt.xticks(range(len(precision)))
plt.bar(range(len(precision)), precision, label='Precision')
plt.bar(range(len(recall)), -recall, label='-Recall')
plt.grid(axis='y')
plt.xlabel('Category')
plt.ylabel('Metric')
plt.legend()


# In[67]:


v, c = np.unique(y_train_origin, return_counts=True)

plt.figure(figsize=(16, 8))

plt.subplot(211)
plt.xticks(range(len(precision)))
plt.bar(range(len(precision)), precision, label='Precesion')
plt.bar(range(len(v)), -10.*c/np.sum(c), label='-10xProbability')
plt.grid(axis='y')
plt.legend()
plt.xlabel('Category')
plt.ylabel('Metric')

plt.subplot(212)
plt.xticks(range(len(recall)))
plt.bar(range(len(recall)), recall, label='Recall')
plt.bar(range(len(v)), -10.*c/np.sum(c), label='-10xProbability')
plt.grid(axis='y')
plt.legend()
plt.xlabel('Category')
plt.ylabel('Metric')


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[44]:


import sys
import os
import matplotlib.image as img

images_dir = 'images'
X, y = (np.vstack(x) 
        for x in zip(*((
            np.expand_dims(img.imread(os.path.join(images_dir, f)), 0),
            int(os.path.splitext(f)[0].split('_')[1]))
            for f in os.listdir(images_dir) 
            if f.startswith('sign'))))
y = np.squeeze(y)

plot_grid(X, range(len(X)), 3, image_size=(2, 2))
X = preprocess_transformation(X)
plot_grid(X, range(len(X)), 3, image_size=(2, 2))
n_channels = X.shape[-1]


# ### Predict the Sign Type for Each Image

# In[45]:


tf.reset_default_graph()

model = create_model(SPEC)
top5_op = topK_operation(model, 5)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, MODEL_FILE)
    p, pi = topK(top5_op, model, X)


# ### Analyze Performance

# In[46]:


tf.reset_default_graph()

model = create_model(SPEC)
accuracy_op = accuracy_operation(model)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, MODEL_FILE)
    print("Accuracy =", evaluate(accuracy_op, model, X, y))


# In[103]:


for i, v in enumerate(y):
    print("expected - {} ({});\t\tpredicted - {} ({});\t\tprobability = {:.3f}".format(
        v, sign_names[v], pi[i, 0], sign_names[pi[i, 0]], p[i, 0]
    ))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[116]:


for i, v in enumerate(y):
    print("image {}: {} -> {}".format(
            i+1, v, ", ".join("{:2} ({:.3f})".format(pi[i,j], p[i,j]) 
                              for j in range(len(p[i])))))


# In[115]:


for i, v in enumerate(y):
    plot_probabilities(X[i].squeeze(), v, p[i], pi[i], sign_names)


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[48]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that 
#                represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, 
#                     by default matplot sets min and max to the actual min and max 
#                     values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, 
#          just extend the plt number for each new feature map entry

def outputFeatureMap(model, image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    _, x, _, _ = model
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    image_input = np.expand_dims(image_input, 0)
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside
    # a function    
    activation = tf_activation.eval(feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(16, 16))
    for featuremap in range(featuremaps):
        plt.subplot(16, 16, featuremap+1) # sets the number of feature maps to show on each row and column
        #plt.title('FM ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], 
                       interpolation="nearest", 
                       vmin=activation_min, 
                       vmax=activation_max, 
                       cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], 
                       interpolation="nearest", 
                       vmax=activation_max, 
                       cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0,:,:, featuremap], 
                       interpolation="nearest", 
                       vmin=activation_min, 
                       cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], 
                       interpolation="nearest", 
                       cmap="gray")
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0.03, hspace=0.03)


# In[66]:


tf.reset_default_graph()

model = create_model(SPEC)
saver = tf.train.Saver()
graph = tf.get_default_graph()

with tf.Session() as sess:
    saver.restore(sess, MODEL_FILE)
    for i in (0, 2):
        for j,tensor in enumerate(tensor 
                                  for x in graph.get_operations() 
                                  if x.name.startswith('Relu')
                                  for tensor in (graph.get_tensor_by_name(x.name + ":0"),)
                                  if (len(tensor.shape) == 4 and 
                                      int(tensor.shape[1]) >= X.shape[1]/4.)):
            outputFeatureMap(model, X[i], tensor, plt_num=i*len(graph.get_operations())+j)

