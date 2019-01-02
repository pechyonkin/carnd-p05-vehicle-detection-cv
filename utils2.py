# functionality for cars detection project

import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import pickle

from os.path import join
from os import listdir
from glob import glob
from cv2 import imread

from skimage.feature import hog
from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

from ipywidgets import interact, fixed

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from utils import * # functionality from previous project



def get_filepaths(subfolder_path):
    '''
    get list of filenames to be read in memory
    go in subfolder's folders and then in each get filenames
    data/
        subfolder1/
            folder1/
            folder2/
            ...
        subfolder2/
            folder1/
            folder2/
            ...
    '''
    paths = []
    # ignore Max OS X system folders starting with '.'
    folders = [f for f in listdir(subfolder_path) if (f[0] != '.')]
    for folder in folders:
        for fname in glob(join(subfolder_path, folder, '*.png')):
            paths.append(fname)
    print("Found {} files in folders:".format(len(paths)))
    print(folders)
    return paths



def bgr_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



def make_data(subfolder, dump_path=None, img_shape=(64,64,3)):
    '''
    this routine prepares a numpy array with the following
    dimensions: (n_imgs, 64, 64, 3)
    it uses cv2 imread and then converts images to RGB
    '''
    filepaths = get_filepaths(subfolder)
    shape = (len(filepaths), img_shape[0], img_shape[1], img_shape[2])
    result = np.zeros(shape).astype(np.uint8)
    for i, f in enumerate(filepaths):
        result[i] = bgr_rgb(imread(f))
    if dump_path is not None:
        result.dump(dump_path)
        print("Data saved successfully at {}".format(dump_path))
    return result



def summarize(data, label, stats=False):
    '''
    prints basic stats of data
    checks:
        - shape of data
        - dtype of data
        - range of data (important to have (0, 255) range for RGB images)
    '''
    print("Shape of {} data: {}".format(label, data.shape))
    print("dtype of {} data: {}".format(label, data.dtype))
    print("Range of {} data: ({},{})".format(label, data.min(), data.max()))
    if stats:
        print("Mean value of {}: {}".format(label, np.mean(data)))
        print("SD of {}: {}".format(label, np.std(data)))
    print()



# Plotting a square matrix of sample images
def plot_imgs(imgs, figsize=(8,8), rows=6, savepath=None, titles=None, title=None):
    '''
    this routine will plot a square matrix of sample images of given shape
    list of labels can be provided by user, length of which should match
    length of images vector
    '''
    assert len(imgs) % rows == 0, "number of images should be a multiple of 'rows'"
    ncols = len(imgs)//rows
    plot = plt.figure(figsize=figsize)
    if title is not None:
        plot.suptitle(title, size=24)
    for i in range(len(imgs)):
        subplot = plot.add_subplot(rows, ncols, i+1)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i])
        plt.imshow(imgs[i])
    if title is not None:
        plot.tight_layout()
        plot.subplots_adjust(top=0.95)       
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')


        
# This gets some images to the plotter routine.
def get_random_images(data, n):
    '''
    this routine will randomly select n images from iamge data array
    returns a view on original array
    it also returns the list of labels
    '''
    rand_idx = np.random.randint(len(data), size=n)
    labels = ["Image {:04d}".format(idx) for idx in rand_idx]
    return data[rand_idx], labels, rand_idx



# Plot random images.
def plot_random_data(data, figsize=(10,10), rows=6, savepath=None, title=None, show_titles=False):
    '''
    this combines above two routines to produce a plot and even save it for report
    it returns images, labels and indices that are later used for plotting HOG features
    '''
    imgs, labels, rand_idx = get_random_images(data, rows*rows)
    if not show_titles:
        labels=None
    plot_imgs(imgs, figsize=figsize, rows=rows, savepath=savepath, titles=labels, title=title)
    return imgs, labels, rand_idx



# Plot HOG images.
def plot_hog_imgs(imgs, labels, what, hog_channel=0, savepath=None):
    '''
    this routine is for plotting HOG features for the same random images
    for both classes for all color channels
    '''
    hog_imgs = []
    # prepare all hog imgs for plotting
    for img in imgs:
        _, hog_img = hog(img[:,:,hog_channel], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
            block_norm='L2-Hys', visualise=True, transform_sqrt=False, feature_vector=True)
        hog_imgs.append(hog_img)
    # plot the images
    title = "HOG Images of {} for Channel {}".format(what, ['R','G','B'][hog_channel])
    plot_imgs(hog_imgs, figsize=(14,14), rows=int(np.sqrt(len(imgs))), savepath=savepath, titles=labels, title=title)



#Plot each channel separated
# NOTE: code in this cell is taken from:
#    https://github.com/asgunzi/CarND-VehicleDetection/blob/master/Data_Exploration.ipynb
# I made changes to plot images horizontally
# the goal was to understand which color space channel contains what kind of information
def plotChannels(image, cspace ='RBG'):
    if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: 
        feature_image = np.copy(image)

    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(20,10))

    ax1.imshow(feature_image[:,:,0],cmap = 'gray')
    ax2.imshow(feature_image[:,:,1],cmap = 'gray')
    ax3.imshow(feature_image[:,:,2],cmap = 'gray')



# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
#     return rhist, ghist, bhist, bin_centers, hist_features
    return hist_features



# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features



# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys', 
                                  transform_sqrt=False, 
                                  visualise=True, 
                                  feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                       transform_sqrt=False, 
                       visualise=False, 
                       feature_vector=feature_vec)
        return features


    
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='YUV', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    result = np.concatenate(img_features)
    return result.astype(np.float32)



# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(data, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in data:
        
        # use single image features to extract features for individual image of data
        image_features = single_img_features(image)
        features.append(image_features)
        
    # Return list of feature vectors
    return np.array(features)



# NOTE: code adapted from:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def plot(img, savepath=None):
    fig = plt.figure(figsize=(16,12))
    plt.imshow(img)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)



def crop(img, y_crop=(400,660)):
    '''
    this function only keeps the region of interest for detection
    ---
    TO DO: make crop as percentage of image height to allow for
    various resolutions (not necessary for this project)
    '''
    return img[y_crop[0]:y_crop[1],:,:]



def box_size(y_val, horizon=425, y_max=530, min_box=20, max_box=260, height=260):
    '''
    this routine determines the optimal size of scan box depending on the vertical coordinate
    the farther the car from us (e.g. the smaller the vertical coordinate)
    the box size will change proportionally to the coordinate
    '''
    assert y_val >= 0, "Vertical coordinate must be a positive number."
    if y_val <= horizon:
        size = min_box
    else:
        slope = (max_box - min_box) / (y_max - horizon)
        intercept = slope * y_max - max_box
        size = slope * y_val - intercept
    return int(size)



def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img



# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list



def slide_dynamic_window(img, y_limit=(450, 511), horizon=25, y_max=130, min_box=20, max_box=260, height=260, x_overlap=0.5, n_vert_windows=15):
    height, width = img.shape[:2]
    y_vals = np.linspace(y_limit[0], y_limit[1], n_vert_windows).astype(np.int)
    window_list = []
    x_start_stop = [0, width]
    # there will be no vertical overlap, only horizontal
    # each step in vertical dimension will use another window size
    xspan = width
    # iterate through all vertical positions
    for ys in y_vals[:]:
        # compute number of horizontal boxes per scan
        bsize = box_size(ys, horizon=horizon, y_max=y_max, min_box=min_box, max_box=max_box, height=height)
        xy_window = (bsize, bsize)
        nx_pix_per_step = np.int(xy_window[0] * x_overlap)
        nx_buffer = np.int(xy_window[0]*(x_overlap))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step) 
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys - bsize // 2
            endy = starty + bsize
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    return window_list



# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='YUV', 
                    spatial_size=(64, 64), orient=11, 
                    pix_per_cell=16, cell_per_block=2, 
                    hog_channel='ALL',
                    hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], spatial_size)      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img)
#         features = single_img_features(test_img, color_space=color_space, 
#                             orient=orient, pix_per_cell=pix_per_cell, 
#                             cell_per_block=cell_per_block, 
#                             hog_channel=hog_channel, 
#                             hog_feat=hog_feat, spatial_size=(32,32))
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes



def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    heatmap = region_of_interest(heatmap) # !!!
    # Return thresholded map
    return heatmap



def draw_labeled_bboxes(img, labels, only_boxes=False, inplace=True):
    if not only_boxes:
        if inplace:
            # draws on original image
            result = img
        else:
            # draws on a copy
            result = np.copy(img)
    boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)
        # Draw the box on the image
        if not only_boxes: cv2.rectangle(result, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    if only_boxes:
        return boxes
    else:
        return result, boxes


    
def labeled_image(img, cars_boxes, thresh, show_plots=False, savepath=None, inplace=False):
    heat = np.zeros_like(img[:,:,0])
    heat = add_heat(heat, cars_boxes)
    heat = apply_threshold(heat, thresh)

    # display all detection boxes
    test_cars_img = draw_boxes(img, cars_boxes, thick=2)

    labels = label(heat)
    labeled_img, labels_boxes = draw_labeled_bboxes(img, labels, inplace=inplace)
    
    if show_plots:
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, figsize=(20,20))

        ax1.imshow(img)
        ax1.set_title('Original Image Segment', size=24)
        ax2.imshow(test_cars_img)
        ax2.set_title('Detected Boxes', size=24)
        ax3.imshow((heat / heat.max() * 255).astype(np.uint8), cmap='hot')
        ax3.set_title('Heatmap of Detected Boxes', size=24)
        ax4.imshow(labeled_img)
        ax4.set_title('Detected Boxes After Thresholding and Labeling', size=24)

        if savepath is not None: fig.savefig(savepath, bbox_inches='tight')
        
    return labeled_img, labels_boxes



