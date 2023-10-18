'''
Common imports and functions to call in other programs
'''

import glob, os, sys, getopt, warnings, cv2, abel
from datetime import datetime
import numpy as np
import pandas as pd
import scipy as sc
from PIL import Image
from scipy import ndimage
from scipy.special import erfc
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from tqdm import tqdm as progress
import imutils
from imutils import resize
import h5py as h5

# directly import commonly used numpy functions
from numpy import sqrt, pi, log, exp, sin
from numpy.fft import fft, fft2, ifft, ifft2

# import units and constants from astropy
import astropy.units as u
from astropy.constants import c, m_e, e, hbar, eps0, a0, k_B, m_p, mu0
from astropy.units import m, um, deg, eV, cm, W, nm, rad, s, ps, fs, kg, mm
from astropy.units import dimensionless_unscaled as dl
e = e.to(u.C)

# set up matplotlib to use desired stylesheet
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
from numpy.fft import fftshift, ifftshift

plt.style.reload_library()
plt.style.use('./atf.mplstyle')

import subprocess
def get_git_commit():
    '''
    Return the current git commit number for the analysis code being run
    '''
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

# Define commonly used colorbar function to improve colorbar spacing in matplotlib
def colorbar(mappable):
    '''
    Fix matplotlib colorbar to be closer to the plot and match the height correctly
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def get_faraday_shot_fnames(data_dir, shot_number):
    '''
    Get the filenames for the raw data of the Faraday rotation diagnostic for a given shot number.
    Returns:
        fname_1 (str): Data filename for camera 1 (SN: 23163771)
        fname_2 (str): Data filename for camera 2 (SN: 23178278)
        ref_fname_1 (str): Reference filename for camera 1 (SN: 23163771)
        ref_fname_2 (str): Reference filename for camera 2 (SN: 23178278)
    '''
    try:
        fname_1 = glob.glob(data_dir + str(shot_number) + '_*cam1.tiff')[0]
        fname_2 = glob.glob(data_dir + str(shot_number) + '_*cam2.tiff')[0]
    except:
        raise FileNotFoundError(f'No file found for shot {shot_number} in directory \'{data_dir}\'')
        
    ref_dir = data_dir + 'background/'
    ref_files = glob.glob(ref_dir + '/*cam1.tiff')
    ref_files.sort(key=lambda x: os.path.getmtime(x) - os.path.getmtime(fname_1))
    ref_fname_1 = ref_files[-1]
    for i, file in enumerate(ref_files):
        if int(data_dir.split('202308')[1][:2]) > 17:
            '''
            On the 18th the camera acquisition script changed to allow for a double trigger signal.
            This means the cameras and gas jet fired twice for each CO2 trigger, and so each shot has
            a reference shot automatically acquired immediately after the actual shot.
            
            The reference file was created on the next TiSa pulse (1.5 Hz) --> 667ms after the shot
            '''
            if abs(os.path.getmtime(file) - os.path.getmtime(fname_1)) < 1:
                # If the reference file was created within 1 second of the shot, it's the right file
                ref_fname_1 = ref_files[i]
                break
        elif os.path.getmtime(file) >  os.path.getmtime(fname_1):
            ref_fname_1 = ref_files[i-1]
            break
    
    ref_fname_2 = ref_fname_1.replace('_cam1', '_cam2')

    return fname_1, fname_2, ref_fname_1, ref_fname_2

def get_faraday_shot(fname_1, fname_2, ref_fname_1, ref_fname_2, plot=False):
    '''
    Get the raw data (as 2d numpy arrays) for the Faraday cameras for a given shot's filenames (handles rotations for different dates)
    '''
    
    img1 = np.array(Image.open(fname_1), dtype=np.float32)/np.float32(2**16)
    img2 = np.array(Image.open(fname_2), dtype=np.float32)/np.float32(2**16)
    ref1 = np.array(Image.open(ref_fname_1), dtype=np.float32)/np.float32(2**16)
    ref2 = np.array(Image.open(ref_fname_2), dtype=np.float32)/np.float32(2**16)
    
    if int(fname_1.split('202308')[1][:2]) < 20:
        # From the 20th forward camera 2 images were rotated before saving
        img2 = np.flip(img2)
        ref2 = np.flip(ref2)
        
    if plot:
        vmax=max(img1.max(), img2.max(), ref1.max(), ref2.max())
        vmin=min(img1.min(), img2.min(), ref1.min(), ref2.min())
        plt.subplot(221),plt.imshow(ref1,cmap = 'gray', vmin=vmin, vmax=vmax)
        plt.title('Camera 1 Background'), plt.xticks([]), plt.yticks([])        
        plt.subplot(222),plt.imshow(img1,cmap = 'gray', vmin=vmin, vmax=vmax)
        plt.title('Camera 1 Data'), plt.xticks([]), plt.yticks([])
        plt.subplot(223),plt.imshow(ref2,cmap = 'gray', vmin=vmin, vmax=vmax)
        plt.title('Camera 2 Background'), plt.xticks([]), plt.yticks([])
        plt.subplot(224),plt.imshow(img2,cmap = 'gray', vmin=vmin, vmax=vmax)
        plt.title('Camera 2 Data'), plt.xticks([]), plt.yticks([])
        plt.show()
    return img1, img2, ref1, ref2