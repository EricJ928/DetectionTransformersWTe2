"""
Created on 23-Feb-2022
@author: Eric Junxiang Jia
# Copyright (c) 2022 by Eric Junxiang Jia <ericjia928@gmail.com>. All Rights Reserved.

This script performs the following pre_process for deep learning training:
1. read a Nanonis .sxm file (topograpy Z channel).
2. linear fit correction to the image.
3. normalize the image.
4. segment large image into a few small images with certain size.

### The two scripts imported for .sxm file extraction and 2d plane fits (SXM.py & SPM.py with a few small modifications)
### origin from an open source library "pySPM" under Apache License, Version 2.0. Copyright 2018 Olivier Scholder <o.scholder@gmail.com>
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import sys
import SXM
from SPM import SPM_image


class NanonisSXM:

    def __init__(self, fname, slice_size=(20, 20), output=False, img_pixsize=None, skip_small_img=False, skip_lowRes_img=False):
        self.fname = fname
        self.load = SXM.SXM(fname)
        self.header = self.load.header
        self.pixel_x = self.load.size['pixels']['x']
        self.pixel_y = self.load.size['pixels']['y']
        self.len_x = self.load.size['real']['x'] * 1e9
        self.len_y = self.load.size['real']['y'] * 1e9
        self.scan_dir = self.load.header['SCAN_DIR'][0][0]
        self.size = dict(pixels={
                'x': int(self.load.header['SCAN_PIXELS'][0][0]),
                'y': int(self.load.header['SCAN_PIXELS'][0][1])
            },real={
                'x': float(self.load.header['SCAN_RANGE'][0][0]),
                'y': float(self.load.header['SCAN_RANGE'][0][1]),
                'unit': 'm'
            })
        for x in self.load.header['DATA_INFO'][1:]:
            if x[1] == 'Z':
                self.zscale = x[2]
                break
        self.data_raw = []
        self.data_raw_norm = []
        self.data_slice = []
        self.data_slice_norm = []

        # check if the whole image size is smaller than the slice size requrired. If so, skip this image.
        if skip_small_img:
            if slice_size[0] > self.len_x or slice_size[1] > self.len_y:
                return

        # check if the image resolution is lower than the slice resolution requrired. If so, skip this image.
        if skip_lowRes_img:
            img_res_x = self.len_x / self.pixel_x
            img_res_y = self.len_y / self.pixel_y
            slice_res_x = slice_size[0] / img_pixsize[0]
            slice_res_y = slice_size[1] / img_pixsize[1]
            if img_res_x > slice_res_x or img_res_y > slice_res_y:
                return

        if output:
            if img_pixsize == None:
                target_folder = '{}nm_RawPix'.format(slice_size[0])
            else:
                target_folder = '{}nm_{}Pix'.format(slice_size[0], img_pixsize[0])
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            self.get_z_raw(plot=True, save=target_folder)
            self.image_segment(newImg_size=(slice_size[0], slice_size[1]), save=target_folder)
            self.z_norm_nm_offset(data_raw=True, data_slice=True)
            self.output_file(img_pixsize, save=target_folder)

    def get_z_raw(self, plot=False, save=''):
        xx = self.load.get_channel('Z')
        if self.scan_dir == 'up':
            self.data_raw = np.flip(xx, axis=0)
        else:
            self.data_raw = xx

        if plot:
            plt.figure()
            plt.imshow(self.data_raw)
            plt.axis('off')
            plt.tight_layout()
            # plt.show()
            plt.savefig('{}/{}_Original.png'.format(save, self.fname.rstrip('.sxm')), dpi=100)
            plt.close()

    def image_segment(self, newImg_size=(20, 20), save=''):
        """This method cut the original image into a few new ones with newImg_size (nm)"""
        oldImg_pixel_x = self.pixel_x
        oldImg_pixel_y = self.pixel_y
        oldImg_len_x = self.len_x
        oldImg_len_y = self.len_y
        # assert oldImg_len_x > newImg_size[0] and oldImg_len_y > newImg_size[1], 'new image size {} is large than the original image ({}, {}).'.format(newImg_size, oldImg_len_x, oldImg_len_y)
        pixel_dens_x = oldImg_pixel_x/oldImg_len_x
        pixel_dens_y = oldImg_pixel_y/oldImg_len_y
        newImg_pixel_x = int(np.ceil(pixel_dens_x * newImg_size[0]))  # the number of pixels for every image along x.
        newImg_pixel_y = int(np.ceil(pixel_dens_y * newImg_size[1]))
        newImg_num_x = int(np.ceil(oldImg_len_x/newImg_size[0]))  # the number of new images along x.
        newImg_num_y = int(np.ceil(oldImg_len_y/newImg_size[1]))  # the number of new images along y.
        newImg_totalNum = newImg_num_x * newImg_num_y
        if oldImg_len_x <= newImg_size[0] or oldImg_len_y <= newImg_size[1]:
            self.data_slice = self.data_raw.reshape(1, self.data_raw.shape[0], self.data_raw.shape[1])
            plt.figure()
            plt.imshow(self.data_slice[0, :, :])
            plt.axis('off')
            plt.tight_layout()
            # plt.show()
            plt.savefig('{}/{}_Slice.png'.format(save, self.fname.rstrip('.sxm')), dpi=100)
            plt.close()
        else:
            self.data_slice = np.zeros((newImg_totalNum, newImg_pixel_x, newImg_pixel_y))
            for i in range(newImg_num_x):
                for j in range(newImg_num_y):
                    Img_index = i*newImg_num_x + j
                    start_pix_x = int((oldImg_pixel_x-newImg_pixel_x) / (newImg_num_x-1) * i)
                    end_pix_x = start_pix_x + newImg_pixel_x
                    start_pix_y = int((oldImg_pixel_y-newImg_pixel_y) / (newImg_num_y-1) * j)
                    end_pix_y = start_pix_y + newImg_pixel_y
                    self.data_slice[Img_index, :, :] = self.data_raw[start_pix_x:end_pix_x, start_pix_y:end_pix_y]
            plt.figure(figsize=(5, 5))
            for i in range(newImg_totalNum):
                plt.subplot(newImg_num_x, newImg_num_y, i+1)
                plt.imshow(self.data_slice[i, :, :])
                plt.axis('off')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            # plt.show()
            plt.savefig('{}/{}_Slice.png'.format(save, self.fname.rstrip('.sxm')), dpi=100)
            plt.close()

    def z_norm_nm_offset(self, data_raw=True, data_slice=True):
        """This method normlizes Z by multiplying 1e9 (convert to nanometers) and offsets it by its mean value."""
        if data_raw:
            self.data_raw_norm = self.data_raw * 1e9
            self.data_raw_norm -= np.mean(self.data_raw_norm)
        else:
            self.data_raw_norm = np.copy(self.data_raw)
        if data_slice:
            self.data_slice_norm = self.data_slice * 1e9
            for i in range(self.data_slice.shape[0]):
                self.data_slice_norm[i, :, :] -= np.mean(self.data_slice_norm[i, :, :])
        else:
            self.data_slice_norm = np.copy(self.data_slice_norm)

    def output_file(self, img_pixsize=None, save=''):
        """output image as numpy array (.npy file). Discard images containing nan."""
        if not np.isnan(self.data_raw).any():
            np.save('{}/{}_Original.npy'.format(save, self.fname.rstrip('.sxm')), self.data_raw)
        for i in range(self.data_slice_norm.shape[0]):
            if not np.isnan(self.data_slice_norm[i, :, :]).any():
                if not img_pixsize == None:
                    output_norm = cv2.resize(self.data_slice_norm[i, :, :], dsize=img_pixsize, interpolation=cv2.INTER_CUBIC)
                    np.save('{}/{}_{:03}.npy'.format(save, self.fname.rstrip('.sxm'), i+1), output_norm)
                else:
                    np.save('{}/{}_{:03}.npy'.format(save, self.fname.rstrip('.sxm'), i+1), self.data_slice_norm[i, :, :])

    def z_norm_2d_plane(self, data):
        SPM_process = SPM_image(BIN=data,
                              channel='Z',
                              real=self.size['real'],
                              _type='Nanonis SXM',
                              zscale=self.zscale,
                              corr=None)
        # Correct with a global 2d plane.
        new_image = SPM_process.correct_plane(inline=False)
        return new_image.pixels


if __name__ == '__main__':
    dirt = 'WTe2_recognition\WTe2_examples\WTe2-HOPG'
    os.chdir(dirt)
    fnames = glob.glob('*.sxm')
    print('No. files:', len(fnames))
    # print('fnames:', fnames)
    # fname = dirt + 'Au#1-4.5K-W1_tip050.sxm'
    # a = NanonisSXM(fnames[10])
    # a = NanonisSXM(fname)
    # a.get_z_raw(plot=True)
    # a.image_segment(newImg_size=(20, 20))
    # a.z_norm_nm_offset(data_raw=True, data_slice=True)
    # a.output_file()
    # a.z_norm()

    for fn in tqdm(fnames):
        # a = NanonisSXM(fn, slice_size=(60, 60), output=True, img_pixsize=(64, 64), skip_small_img=True, skip_lowRes_img=True)
        a = NanonisSXM(fn, slice_size=(40, 40), output=True, img_pixsize=(64, 64), skip_small_img=True, skip_lowRes_img=True)
    # for i, fn in enumerate(fnames):
    #     print('--> image pre-processing: {}/{}'.format(i+1, len(fnames)))
    #     a = NanonisSXM(fn, slice_size=(40, 40), output=True, img_pixsize=(64, 64))
