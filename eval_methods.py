# -*- coding:utf-8 -*-

import numpy as np

class ImageEvalue(object):

    def image_mean(self, image):
        mean = np.mean(image)
        return mean

    def image_var(self, image, mean):
        m, n = np.shape(image)
        var = np.sqrt(np.sum((image - mean) ** 2) / (m * n - 1))
        return var

    def images_cov(self, image1, image2, mean1, mean2):
        m, n = np.shape(image1)
        cov = np.sum((image1 - mean1) * (image2 - mean2)) / (m * n - 1)
        return cov

    def PSNR(self, O, F):
        '''
        :param O: 原始图像
        :param F: 滤波后的图像
        '''
        MES = np.mean((np.array(O) - np.array(F)) ** 2)
        PSNR = 10 * np.log10(255 ** 2 / MES)
        return PSNR

    def SSIM(self, O, F):
        '''
        :param O: 原始图像
        :param F: 滤波后的图像
        '''
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        meanO = self.image_mean(O)
        meanF = self.image_mean(F)
        varO = self.image_var(O, meanO)
        varF = self.image_var(O, meanF)
        covOF = self.images_cov(O, F, meanO, meanF)
        SSIM = (2 * meanO * meanF + c1) * (2 * covOF + c2) / (
                (meanO ** 2 + meanF ** 2 + c1) * (varO ** 2 + varF ** 2 + c2))
        return SSIM

    def IEF(self, O, F, X):
        '''
        :param O: 原始图像
        :param F: 滤波后的图像
        :param X: 噪声图像
        '''
        IEF = np.sum((X - O) ** 2) / np.sum((F - O) ** 2)
        return IEF

    def UQI(self, O, F):
        '''
        :param O: 原始图像
        :param F: 滤波后的图像
        '''
        meanO = self.image_mean(O)
        meanF = self.image_mean(F)
        varO = self.image_var(O, meanO)
        varF = self.image_var(F, meanF)
        covOF = self.images_cov(O, F, meanO, meanF)
        UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
        return UQI

