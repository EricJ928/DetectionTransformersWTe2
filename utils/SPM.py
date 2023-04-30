# -- coding: utf-8 --

# Copyright 2018 Olivier Scholder <o.scholder@gmail.com>

"""
Library to handle SPM data.
This is the core module of all images retrieved by SPM and ToF-SIMS.

24-Feb-2022
This script has been simplified by EricJ. For the original script, please
visit Olivier Scholder. (2018, November 28). scholi/pySPM: pySPM v0.2.16
(Version v0.2.16). Zenodo. http://doi.org/10.5281/zenodo.998575
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import scipy.optimize
import skimage
import skimage.exposure
import skimage.filters
import scipy.interpolate
from skimage import transform as tf
import copy
import sys
import matplotlib as mpl
import warnings

try:
    from skimage.filters import threshold_local
except:
    # For compatibility with old versions of skimage
    from skimage.filters import threshold_adaptive as threshold_local

class SPM_image:
    """
    Main class to handle SPM images.
    This class contains the pixels data of the images as well as it's real size.
    It also provides a lot of tools to correct and perform various analysis and tasks on the image.
    """

    def __init__(self, BIN, channel='Topography',
                 corr=None, real=None, zscale='?', _type='Unknown'):
        """
        Create a new SPM_image

        Parameters
        ----------
        BIN : 2D numpy array
            The pixel values of the image as a 2D numpy array
        channel : string
            The name of the channel. What does the image represents?
        corr : string or None
            'slope' : correct the SPM image for its slope (see pySPM.SPM.SPM_image.correct_slope)
            'lines' : correct the SPM image for its lines (see pySPM.SPM.SPM_image.correct_lines)
            'plane' : correct the SPM image by plane fitting (see pySPM.SPM.SPM_image.correct_plane)
        real : None or dictionary
            Information about the real size of the image {'x':width,'y':height,'unit':unit_name}
        zscale : string
            Unit used to describe the z-scale. (units of the data of BIN)
        _type : string
            represent the type of measurement
        """
        self.channel = channel
        self.direction = 'Unknown'
        self.size = {'pixels': {'x': BIN.shape[1], 'y': BIN.shape[0]}}
        if not real is None:
            self.size['real'] = real
        else:
            self.size['real'] = {'unit': 'pixels',
                                 'x': BIN.shape[1], 'y': BIN.shape[0]}
        if not 'unit' in self.size['real']:
            self.size['real']['unit'] = 'px'
        self.pixels = BIN
        self.type = _type
        self.zscale = zscale
        if corr is not None:
            if corr.lower() == 'slope':
                self.correct_slope()
            elif corr.lower() == 'lines':
                self.correct_lines()
            elif corr.lower() == 'plane':
                self.correct_plane()

    def correct_slope(self, inline=True):
        """
        Correct the image by subtracting a fitted slope along the y-axis
        """
        s = np.mean(self.pixels, axis=1)
        i = np.arange(len(s))
        fit = np.polyfit(i, s, 1)
        if inline:
            self.pixels -= np.tile(np.polyval(fit, i).reshape(len(i), 1), len(i))
            return self
        else:
            New = copy.deepcopy(self)
            New.pixels -= np.tile(np.polyval(fit, i).reshape(len(i), 1), len(i))
            return New

    def correct_plane(self, inline=True, mask=None):
        """
        Correct the image by subtracting a fitted 2D-plane on the data

        Parameters
        ----------
        inline : bool
            If True the data of the current image will be updated otherwise a new image is created
        mask : None or 2D numpy array
            If not None define on which pixels the data should be taken.
        """
        x = np.arange(self.pixels.shape[1])
        y = np.arange(self.pixels.shape[0])
        X0, Y0 = np.meshgrid(x, y)
        Z0 = self.pixels
        if mask is not None:
            X = X0[mask]
            Y = Y0[mask]
            Z = Z0[mask]
        else:
            X = X0
            Y = Y0
            Z = Z0
        A = np.column_stack((np.ones(Z.ravel().size), X.ravel(), Y.ravel()))
        c, resid, rank, sigma = np.linalg.lstsq(A, Z.ravel(), rcond=-1)
        if inline:
            self.pixels -= c[0] * \
                np.ones(self.pixels.shape) + c[1] * X0 + c[2] * Y0
            return self
        else:
            New = copy.deepcopy(self)
            New.pixels -= c[0]*np.ones(self.pixels.shape) + c[1] * X0 + c[2] * Y0
            return New

    def correct_lines(self, inline=True):
        """
        Subtract the average of each line for the image.

        if inline is True the current data are updated otherwise a new image with the corrected data is returned
        """
        if inline:
            self.pixels -= np.tile(np.mean(self.pixels, axis=1).T, (self.pixels.shape[0], 1)).T
            return self
        else:
            New = copy.deepcopy(self)
            New.pixels -= np.tile(np.mean(self.pixels, axis=1).T, (self.pixels.shape[0], 1)).T
            return New

    def offset(self, profiles, width=1, ax=None, col='w', inline=True, **kargs):
        """
        Correct an image by offsetting each row individually in order that the lines passed as argument in "profiles" becomes flat.

        Parameters
        ----------
        profiles: list of list
            each sublist represent a line as [x1, y1, x2, y2] in pixels known to be flat
        width : int, float
            the line width in pixels used for better statistics
        ax : matplotlib axis or None
            If not None, axis in which the profiles will be plotted in
        inline : bool
            If True perform the correction on the current object, otherwise return a new image
        col : string
            matrplotlib color used to plot the profiles (if ax is not None)
        labels : bool
            display a label number with each profile
        **kargs: arguments passed further to get_row_profile.
            axPixels: set to True if you axis "ax" have the data plotted in pixel instead of real distance

        Example
        -------
        Exampel if the data are plotted in pixels:
        >>> topo = pySPM.SPM_image(...)
        >>> fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        >>> topoC = topo.offset([[150, 0, 220, 255]], inline=False,axPixels=True)
        >>> topo.show(pixels=True, ax=ax[0])
        >>> topoC.show(ax=ax[1]);

        Example if the data are plotted with real units
        >>> topo = pySPM.SPM_image(...)
        >>> fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        >>> topoC = topo.offset([[150, 0, 220, 255]], inline=False)
        >>> topo.show(ax=ax[0])
        >>> topoC.show(ax=ax[1]);
        """
        offset = np.zeros(self.pixels.shape[0])
        counts = np.zeros(self.pixels.shape[0])
        for i, p in enumerate(profiles):
            if kargs.get('labels', False):
                y, D = self.get_row_profile(*p, width=width, ax=ax, col=col, label=str(i), **kargs)
            else:
                y, D = self.get_row_profile(*p, width=width, ax=ax, col=col, **kargs)
            counts[y] += 1
            offset[y[1:]] += np.diff(D)
        counts[counts == 0] = 1
        offset = offset/counts
        offset = np.cumsum(offset)
        offset = offset.reshape((self.pixels.shape[0], 1))
        if inline:
            self.pixels = self.pixels - \
                np.flipud(np.repeat(offset, self.pixels.shape[1], axis=1))
            return self
        else:
            C = copy.deepcopy(self)
            C.pixels = self.pixels - \
                np.flipud(np.repeat(offset, self.pixels.shape[1], axis=1))
            return C


    def correct_median_diff(self, inline=True):
        """
        Correct the image with the median difference
        """
        N = self.pixels
        # Difference of the pixel between two consecutive row
        N2 = N-np.vstack([N[:1, :],N[:-1, :]])
        # Take the median of the difference and cumsum them
        C = np.cumsum(np.median(N2, axis=1))
        # Extend the vector to a matrix (row copy)
        D = np.tile(C, (N.shape[0], 1)).T
        if inline:
            self.pixels = N-D
        else:
            New = copy.deepcopy(self)
            New.pixels = N-D
            return New

    def corr_fit2d(self, nx=2, ny=1, poly=False, inline=True, mask=None):
        """
        Subtract a fitted 2D-polynom of nx and ny order from the data

        Parameters
        ----------
        nx : int
            the polynom order for the x-axis
        ny : int
            the polynom order for the y-axis
        poly : bool
            if True the polynom is returned as output
        inline : bool
            create a new object?
        mask : 2D numpy array
            mask where the fitting should be performed
        """
        r, z = fit2d(self.pixels, nx, ny, mask=mask)
        if inline:
            self.pixels -= z
        else:
            N = copy.deepcopy(self)
            N.pixels -= z
            if poly:
                return N, z
            return N
        if poly:
            return z
        return self

    def zero_min(self, inline=True):
        """
        Shift the values so that the minimum becomes zero.
        """
        if inline:
            self.pixels -= np.min(self.pixels)
            return self
        else:
            N = copy.deepcopy(self)
            N.pixels -= np.min(N.pixels)
            return N

def normalize(data, sig=None, vmin=None, vmax=None):
    """
    Normalize the input data. Minimum_value -> 0 and maximum_value -> 1

    Parameters
    ----------
    data : numpy array
        input data
    sig : float or None
        if not None:
            mean(data)-sig*standard_deviation(data) -> 0
            mean(data)+sig*standard_deviation(data) -> 1
    vmin : float or None
        if not None, define the lower bound i.e.  vmin -> 0
    vmax : float or None
        if not None, defines the upper bound i.e. vmax -> 0

    Note
    ----
    All values below the lower bound will be = 0
    and all values above the upper bound will be = 1


    """
    if sig is None:
        mi = np.min(data)
        ma = np.max(data)
    else:
        s = sig*np.std(data)
        mi = np.mean(data)-s
        ma = np.mean(data)+s
    if vmin is not None:
        mi = vmin
    if vmax is not None:
        ma = vmax
    N = (data-mi)/(ma-mi)
    N[N < 0] = 0
    N[N > 1] = 1
    return N

def imshow_sig(img, sig=1, ax=None, **kargs):
    """
    Shortcut to plot a numpy array around it's mean with bounds ±sig sigmas

    Parameters
    ----------
    img : 2D numpy array
        input image to display
    sig : float
        The number of standard-deviation to plot
    ax : matplotlib axis
        matplotlib axis to use. If None, the current axis (plt.gca() will be used).
    **kargs : additional parameters
        will be passed to the imshow function of matplotls2 = pySPM.Nanoscan("%s/CyI5b_PCB_ns.xml"%(Path))ib
    """
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    std = np.std(img)
    avg = np.mean(img)
    vmin = avg - sig * std
    vmax = avg + sig * std
    ax.imshow(img, vmin=vmin, vmax=vmax, **kargs)

def adjust_position(fixed, to_adjust, shift=False):
    """ Shift the current pixels to match a fixed image by rolling the data"""
    adj = copy.deepcopy(to_adjust)
    cor = np.fft.fft2(fixed)
    cor = np.abs(np.fft.ifft2(np.conj(cor) * np.fft.fft2(to_adjust)))
    cor = cor / to_adjust.size
    ypeak, xpeak = np.unravel_index(cor.argmax(), cor.shape)
    shift = [-(ypeak-1), -(xpeak-1)]
    adj = np.roll(to_adjust, shift[0], axis=0)
    adj = np.roll(adj, shift[1], axis=1)
    if shift:
        return adj, shift
    return adj


def tukeyfy(A, alpha, type='default'):
    """
    Apply a Tukey window on the current image

    Parameters
    ----------
    A : 2D numpy array
        input array
    alpha : float
        Size of the Tukey windows in percent of the image (≥0 and ≤1)
    type : string
        if not "default" perform a mean centering (data will blend down to its mean instead of 0)
    """
    tuky = tukeywin(A.shape[0], alpha)
    tukx = tukeywin(A.shape[1], alpha)
    tuk = np.multiply(tukx[:, None].T, tuky[:, None])
    if type is 'default':
        return A * tuk
    avg = np.mean(A)
    return avg+(A-avg) * tuk

def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.

    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output

    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html

    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length)  # rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x < alpha/2
    w[first_condition] = 0.5 * \
        (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2)))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x >= (1 - alpha/2)
    w[third_condition] = 0.5 * \
        (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))
    return w

def overlay(ax, mask, color, **kargs):
    """
    Plot an overlay on an existing axis

    Parameters
    ----------
    ax : matplotlib axis
        input axis
    mask : 2D numpy array
        Binary array where a mask should be plotted
    color : string
        The color of the mask to  plot
    **kargs: additional parameters
        passed to the imshow function of matploltib
    """
    m = ma.masked_array(mask, ~mask)
    col = np.array(colors.colorConverter.to_rgba(color))
    I = col[:, None, None].T*m[:, :, None]
    ax.imshow(I, **kargs)

def normP(x, p, trunk=True):
    """
    Normalize the input data accroding to its percentile value.

    Parameters
    ----------
    x : 2D numpy array
        input data
    p : float
        percentile to normalize the data.
        lower bound = p percentile
        upper bound = (100-p) percentile
    trunk : bool
        If True the data are truncated between 0 and 1
    """
    thresh_high = np.percentile(x, 100-p)
    thresh_low = np.percentile(x, p)
    if thresh_low == thresh_high:
        thresh_high = np.max(x)
        thresh_low = np.min(x)
    if thresh_low == thresh_high:
        thresh_high = thresh_low+1
    r = (x-thresh_low)/(thresh_high-thresh_low)
    if trunk:
        r[r < 0] = 0
        r[r > 1] = 1
    return r

def beam_profile(target, source, mu=1e-6, tukey=0, meanCorr=False, source_tukey=None, real=np.abs, **kargs):
    """
    Calculate the PSF by deconvolution of the target
    with the source using a Tikhonov regularization of factor mu.
    """
    if source_tukey is None:
        source_tukey = tukey
    if kargs.get('source_centering', False):
        source = 2*source-1
    if meanCorr:
        target = target-np.mean(target)
    if tukey>0:
        target = tukeyfy(target, tukey)
    if source_tukey>0:
        source = tukeyfy(source, tukey)
    tf = np.fft.fft2(source)
    tf /= np.size(tf)
    recon_tf = np.conj(tf) / (np.abs(tf)**2 + mu)
    return np.fft.fftshift(real(np.fft.ifft2(np.fft.fft2(target) * recon_tf)))/np.size(target)

def beam_profile1d(target, source, mu=1e-6, real=np.abs):
    source = source
    tf = np.fft.fft(source)
    tf /= np.size(tf)
    recon_tf = np.conj(tf) / (np.abs(tf)**2 + mu)
    F = np.fft.fft(target) * recon_tf
    return np.fft.fftshift(real(np.fft.ifft(F))), F

def zoom_center(img, sx, sy=None):
    """
    Zoom by taking the sx × sy central pixels

    Parameters
    ----------
    img : 2D numpy array
        The input data
    sx : int
        The number of pixels along the x-axis to take

    sy : int or None
        The number of pixels alongs the y-axis to take.
        If None take the same value as for sx
    """
    if sy is None:
        sy = sx
    assert type(sx) is int
    assert type(sy) is int
    return img[
        img.shape[0]//2-sy//2: img.shape[0]//2 + sy//2,
        img.shape[1]//2-sx//2: img.shape[1]//2 + sx//2]

def px2real(x, y, size, ext):
    rx = ext[0]+(x/size[1])*(ext[1]-ext[0])
    ry = ext[2]+(y/size[0])*(ext[3]-ext[2])
    return rx, ry


def real2px(x, y, size, ext):
    px = size[1]*(x-ext[0])/(ext[1]-ext[0])
    py = size[0]*(y-ext[2])/(ext[3]-ext[2])
    return px, py

def fit2d(Z0, dx=2, dy=1, mask=None):
    """
    Fit the input data with a 2D polynom of order dx × dy

    Parameters
    ----------
    Z0 : 2D numpy array
        input data
    dx : int
        order of the polynom for the x-axis
    dy : int
        order of the polynom for the y-xis
    mask : 2D numpy array
        Give a mask where True values only will be used to perform the fitting

    Returns
    -------
    numpy array
        fitting parameters
    2D numpy array
        result of the polynom
    """
    x = np.arange(Z0.shape[1], dtype=np.float)
    y = np.arange(Z0.shape[0], dtype=np.float)
    X0, Y0 = np.meshgrid(x, y)
    if mask is not None:
        X = X0[mask]
        Y = Y0[mask]
        Z = Z0[mask]
    else:
        X = X0
        Y = Y0
        Z = Z0
    x2 = X.ravel()
    y2 = Y.ravel()
    A = np.vstack([x2**i for i in range(dx+1)])
    A = np.vstack([A]+[y2**i for i in range(1, dy+1)])
    res = scipy.optimize.lsq_linear(A.T, Z.ravel())
    r = res['x']
    Z2 = r[0]*np.ones(Z0.shape)
    for i in range(1, dx+1):
        Z2 += r[i]*(X0**i)
    for i in range(1, dy+1):
        Z2 += r[dx+i]*(Y0**i)
    return r, Z2


def warp_and_cut(img, tform, cut=True):
    """
    Perform an Affine transform on the input data and cut them if cut=True

    Parameters
    ----------
    img : 2D numpy array
        input data
    tform : skimage.transform
        An Affine fransform to perform on the data
    cut : bool
        Should the data be cutted?
    """
    New = tf.warp(img, tform, preserve_range=True)
    Cut = [0, 0] + list(img.shape)
    if tform.translation[0] >= 0:
        Cut[2] -= tform.translation[0]
    elif tform.translation[0] < 0:
        Cut[0] -= tform.translation[0]
    if tform.translation[1] >= 0:
        Cut[1] += tform.translation[1]
    elif tform.translation[1] < 0:
        Cut[3] += tform.translation[1]
    Cut = [int(x) for x in Cut]
    if cut:
        New = cut(New, Cut)
    return New, Cut

def get_profile(I, x1, y1, x2, y2, width=0, ax=None, color='w', alpha=0, N=None,\
        transx=lambda x: x, transy=lambda x: x, interp_order=1, **kargs):
    """
    Get a profile from an input matrix.
    Low-level function. Doc will come laters2 = pySPM.Nanoscan("%s/CyI5b_PCB_ns.xml"%(Path))
    """
    d = np.sqrt((x2-x1)**2+(y2-y1)**2)
    if N is None:
        N = int(d)+1
    P = []
    dx = -width/2*(y2-y1)/d
    dy = width/2*(x2-x1)/d
    for w in np.linspace(-width/2, width/2, max(1,width)):
        dx = -w*(y2-y1)/d
        dy = w*(x2-x1)/d
        x = np.linspace(x1+dx, x2+dx, N)
        y = np.linspace(y1+dy, y2+dy, N)
        M = scipy.ndimage.interpolation.map_coordinates(I, np.vstack((y, x)), order=interp_order)
        P.append(M)
    if kargs.get('debug',False):
        print("get_profile input coordinates:",x1,y1,x2,y2)
    if not ax is None:
        x1 = transx(x1)
        x2 = transx(x2)
        y1 = transy(y1)
        y2 = transy(y2)
        if kargs.get('debug',False):
            print("Drawing coordinates:",x1,y1,x2,y2)
        dx = -width/2*(y2-y1)/d
        dy = width/2*(x2-x1)/d
        if type(color) in [tuple, list]:
            ax.plot([x1, x2], [y1, y2], color=color, alpha=kargs.get('linealpha',1))
            ax.plot([x1-dx, x1+dx], [y1-dy, y1+dy], color=color, alpha=kargs.get('linealpha',1))
            ax.plot([x2-dx, x2+dx], [y2-dy, y2+dy], color=color, alpha=kargs.get('linealpha',1))
        else:
            ax.plot([x1, x2], [y1, y2], color, alpha=kargs.get('linealpha',1), lw=kargs.get('lw',1))
            ax.plot([x1-dx, x1+dx], [y1-dy, y1+dy], color, alpha=kargs.get('linealpha',1))
            ax.plot([x2-dx, x2+dx], [y2-dy, y2+dy], color, alpha=kargs.get('linealpha',1))
        if alpha>0:
            import matplotlib.patches
            ax.add_patch(matplotlib.patches.Rectangle(
                (x1+dx,y1+dy),
                2*np.sqrt(dx**2+dy**2),
                np.sqrt((x2-x1)**2+(y2-y1)**2),
                -np.degrees(np.arctan2(x2-x1,y2-y1)), color=color, alpha=alpha))
    if len(P)==1:
        return np.linspace(0, d, N), P[0]
    return np.linspace(0, d, N), np.vstack(P).T


def dist_v2(img, dx=1, dy=1):
    """
    Return a 2D array with the distance in pixel with the clothest corner of the array.
    """
    x2 = np.arange(img.shape[1])
    x2 = (np.minimum(x2, img.shape[1]-x2) * dx)**2
    y2 = np.arange(img.shape[0])
    y2 = (np.minimum(y2, img.shape[0] - y2) * dy)**2
    X, Y = np.meshgrid(x2, y2)
    return np.sqrt(X+Y)

def generate_k_matrices(A, dx, dy):
    """
    GENERATE_K_MATRICES k-Matrix generation (helper function).
    generates k-matrices for the 2D-channel CHANNEL.

    K is a matrix of the same size as the pixel matrix A, containing the real-life frequency distance of each
    pixel position to the nearest corner of an matrix that is one pixel
    wider/higher.
    KX is of the same size as K and contains the real-life  difference in x-direction of each pixel position to the nearest corner
    of a matrix that is one pixel wider/higher.
    Similarly, KY is of the  same size as K, containing the real-life difference in y-direction of  each pixel position to the nearest corner of an matrix that is one
    pixel wider/higher.
    """
    ny, nx = A.shape
    dkx = 2*np.pi/(nx*dx)
    dky = 2*np.pi/(ny*dy)

    ky = np.arange(0, ny);
    ky = (np.mod(ky+ny/2, ny) - ny/2) * dky

    kx = np.arange(0, nx);
    kx = (np.mod(kx+nx/2, nx) - nx/2) * dkx

    kx, ky = np.meshgrid(kx, ky)
    k = dist_v2(A, dkx, dky)
    k[0, 0] = 1.0 # Prevent division by zero error and illegal operand errors. This may be improved...
    return k, kx, ky

def mfm_tf(nx, dx, ny, dy, tf_in, derivative=0, transform=0, z=0, A=0, theta=None, phi=None, d=None, delta_w=None):
    """
    Draft for the MFM tf function
    """
    k, kx, ky = generate_k_matrices(tf_in, dx, dy)
    # Distance loss
    tf_out = np.exp(-z*k)
    if d is not None:
        tf_out = tf_out / 2.0
        if not np.isinf(d):
            if d == 0:
                tf_out *= k
            else:
                tf_out *= 1 - np.exp(-d*k)
    if A == 0:
        if transform != 0:
            assert theta is not None
            assert phi is not None
            tf_out *= ((np.cos(theta)+1j*(np.cos(phi)*np.sin(-theta)*kx+np.sin(phi)*np.sin(-theta)*ky)) / k)**transform
        if derivative == 1:
            tf_out *= k
    else:
        pass # TODO
    return tf_out * tf_in

def mfm_inv_calc_flat(img, z, tf_in, thickness=None, delta_w=None, amplitude=0, derivative=0, transform=0, mu=1e-8):
    """
    MFM inv calc function
    """
    theta = np.radians(12)
    phi = np.radians(-90)
    ny, nx = img.shape
    tf = mfm_tf(nx, 1, ny, 1, tf_in, derivative, transform, z, amplitude, theta, phi, thickness, delta_w)
    tf[0,0] = np.real(np.mean(tf))
    recon_tf = np.conj(tf) / (mu+np.abs(tf)**2)
    work_img = np.abs(np.fft.ifft2(np.fft.fft2(img) * recon_tf ))
    return work_img

def get_tik_tf(Img, mu, tukey=0, source_tukey=0, debug=False, d=200, real=np.real):
    import scipy
    def fit(x, a ,A, bg, x0):
        return bg+(A-bg)*np.exp(-abs(x-x0)/a)

    x = np.arange(Img.shape[1])
    y = np.arange(Img.shape[0])
    X, Y = np.meshgrid(x, y)
    x0 = Img.shape[1]/2
    y0 = Img.shape[0]/2
    R = np.sqrt((X-x0)**2+(Y-y0)**2)

    Z = beam_profile(Img, Img, mu=mu, tukey=tukey, source_tukey=source_tukey, real=real)
    zoom = zoom_center(Z, d)
    P = zoom[zoom.shape[0]//2, :]
    p0 = (1,np.max(zoom), 0, len(P)/2)
    popt, pcov = scipy.optimize.curve_fit(fit, np.arange(len(P)), P, p0, bounds=((0,0,-np.inf,0),np.inf))
    bg = popt[2]
    a = popt[0]
    if debug:
        return bg+np.exp(-np.abs(R)/a), Z, p0, popt
    return bg+np.exp(-np.abs(R)/a)
