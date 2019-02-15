import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

def degrees2radians(x):
    """ convert degrees to radians """
    return x * np.pi/180 

def ellipse2points(e):
    """
    converts an ellipse in (cx, cy), (ma, MA), theta form into a three 2D-point array (center, minor axis, and major axis 
    xy coordinates)
    INPUTS:
    ----------
    e: [(cx, cy), (ma, MA), theta] (openCV ellipse)
        center coordinates (float), minor and major axes (float), angle of rotation (float--degrees)
    OUTSPUTS:
    ----------
    c: float 
        center xy coordinates
    pMA: 2D-float
        Major axis xy coordinates
    pma: 2D-float
        Minor axis xy coordinates
    """
    c = e[0] # center
    MA, ma = e[1] # minor and major axis lengths
    theta = e[2] # orientation of the ellipse
    dd, dc = e[3]

    resolution = MA * ma
    MA = MA/2
    ma = ma/2
    
    if dd > 0:
        theta_ad = theta
        pMA = (c[0] + MA*np.cos(degrees2radians(theta_ad)), c[1] - MA*np.sin(degrees2radians(theta_ad))) # Major axis point
    else:
        theta_ad = theta+180
        pMA = (c[0] + MA*np.cos(degrees2radians(theta_ad)), c[1] - MA*np.sin(degrees2radians(theta_ad))) # Major axis point 

    if dd*dc > 0:
        pma = (c[0] + ma*np.cos(degrees2radians(theta+90)), c[1] - ma*np.sin(degrees2radians(theta+90))) # minor axis point
    else:
        pma = (c[0] + ma*np.cos(degrees2radians(theta-90)), c[1] - ma*np.sin(degrees2radians(theta-90))) # minor axis point

    return np.array([c, pMA, pma], dtype=np.float32), (theta_ad, dc, resolution)

def compute_affine(p1, p2):
    """
    computes an affine transform between 3-corresponding points in each image
    
    INPUTS:
    ---------
    p1: numpy array with dimensions 3x2
        three, 2-D points in first image
    p2: numpy array with dimensions 3x2
        corresponding three, 2-D points in second image to compute affine transform
    OUTPUTS:
    ----------
    T: numpy array with dimension 2x3
        affine transform matrix
    """
    T = cv2.getAffineTransform(p1, p2)
    return T


def histogram_match(im1, im2):
    """
    Computes a new image from im1 with matching histogram to im2.
    Performs histogram matching by computing a mapping function from the cdf of im1 to the cdf of im2
    
    INPUTS:
    ---------
    im1: numpy array
        input image
    im2: numpy array
        target histogram image
    OUTPUTS:
    ---------
    im_matched: numpy array
        image of shape im1 with augmented histogram matching im2
    """
    # compute image histograms
    h1, bins1 = np.histogram(im1, bins=256, range=[0,255])
    h2, bins2 = np.histogram(im2, bins=256, range=[0,255])
    # compute cummulative distribution funcitons
    cdf1 = h1.cumsum()
    cdf2 = h2.cumsum()
    # compute mapping
    g_out = [np.argmin(np.abs(cdf2 - f)) for f in cdf1]  
    # map im1 grey values to match
    im_matched = np.array(g_out)[im1]
    
    return im_matched

