import numpy as np


def subtract_2d_plane(img):
    """
    Subtract a fitted 2d plane from the input image.

    Parameters:     img : rank2 numpy array with the same shape [0] and [1]
                        Input image.
    Return:         img_sub : rank2 numpy array
                        New image after substraction.
    """

    m = img.shape[0]
    X1, X2 = np.mgrid[:m, :m]

    # Regression
    X = np.hstack((np.reshape(X1, (m*m, 1)), np.reshape(X2, (m*m, 1))))
    X = np.hstack((np.ones((m*m, 1)), X))

    YY = np.reshape(img, (m*m, 1))
    theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (m, m))

    # Subtraction
    img_sub = img - plane

    return img_sub


def img_norm(img):
    """
    Normalize the input image to 0~1.

    Parameters:     img : rank2 numpy array with the same shape [0] and [1]
                        Input image.
    Return:         img_norm : rank2 numpy array
                        New image after normalization.
    """

    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = (img-img_min) / (img_max-img_min)

    return img_norm


def data_augmentation(img):
    """
    Data augmentation of the input image.

    Parameters:     img : rank2 numpy array with the same shape [0] and [1]
                        Input image.
    Return:         img_aug : rank3 numpy array
                        Data augmentation of 8 new images (mirror and/or rotate by 90/180/270 degrees).
    """
    img_aug = np.zeros((8, img.shape[0], img.shape[1]))
    img_aug[0, :, :] = img
    img_aug[1, :, :] = np.rot90(img, 1)
    img_aug[2, :, :] = np.rot90(img, 2)
    img_aug[3, :, :] = np.rot90(img, 3)
    img_aug[4, :, :] = np.flipud(img)
    img_aug[5, :, :] = np.flipud(img_aug[1, :, :])
    img_aug[6, :, :] = np.flipud(img_aug[2, :, :])
    img_aug[7, :, :] = np.flipud(img_aug[3, :, :])

    return img_aug
