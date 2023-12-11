
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import glob
import os
import visualPercepUtils as vpu
import matplotlib.pyplot as plt



def saveImg(image, path, filename, extension):
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image.save(os.path.join(path, filename + extension))


def multiHist(im, n, bins=256):
    def getQuadrants(image, level):
        if level == 0:
            hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 256))
            return [hist]
        
        h, w = image.shape
        h2, w2 = h // 2, w // 2
        quadrants = [
            image[:h2, :w2], image[:h2, w2:],
            image[h2:, :w2], image[h2:, w2:]
        ]

        histograms = []
        for quad in quadrants:
            histograms.extend(getQuadrants(quad, level - 1))

        return histograms

    return getQuadrants(im, n - 1)

def checkBoardImg(im, m, n):
    h, w = im.shape[:2]
    out_img = np.copy(im)
    cell_h, cell_w = h // m, w // n

    for i in range(m):
        for j in range(n):
            if (i + j) % 2 == 1:
                if len(im.shape) == 3:  # Для цветных изображений
                    out_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = 255 - out_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :]
                else:  # Для изображений в оттенках серого
                    out_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = 255 - out_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
    return out_img


def testCheckBoard():
    imfile = './1imgs-P2/1.png'  # конкретный файл изображения
    im = np.array(Image.open(imfile).convert('L'))
    histograms = multiHist(im, n=3)  # Вызов multiHist с n=3

    # Отображение гистограмм
    for hist in histograms:
        plt.figure()
        plt.bar(range(len(hist)), hist)
        plt.show()
    
    check_board_img = checkBoardImg(im, 4, 4)
    vpu.showInGrid([im, check_board_img], m=1, n=2, title="Original and CheckBoard Image")
    saveImg(check_board_img, './1imgs-out-P2/', 'check_board_image', '.png')

testCheckBoard()

def expTransf(alpha, n, l0, l1, bInc):
    l = np.linspace(l0, l1, n)
    
    # Применяем экспоненциальное преобразование
    T_l = np.exp(-alpha * (l - l.mean())**2)

    # Нормализация результатов, чтобы они находились в диапазоне [l0, l1]
    T_l = (T_l - T_l.min()) / (T_l.max() - T_l.min()) * (l1 - l0) + l0

    if not bInc:
        T_l = np.flip(T_l)  # Инвертируем, если bInc=False
    
    return T_l

# Пример использования
alpha = 0.001
n = 256
l0, l1 = 0, 255
bInc = True

transf_func = expTransf(alpha, n, l0, l1, bInc)
plt.plot(transf_func)
plt.title("Exponential Transformation Function")
plt.xlabel("Input Gray Level")
plt.ylabel("Output Gray Level")
plt.show()

def transfImage(im, f):
    # Применяем функцию преобразования к каждому пикселю
    transformed_im = np.interp(im, np.linspace(l0, l1, n), f).astype(im.dtype)
    return transformed_im

# Пример использования на изображении
im = np.array(Image.open('./1imgs-P2/1.png').convert('L'))  # Загрузка и конвертация в оттенки серого
transformed_im = transfImage(im, transf_func)

plt.imshow(transformed_im, cmap='gray')
plt.title("Transformed Image")
plt.show()

def histeq(im, nbins=256):
    imhist, bins = np.histogram(im.flatten(), list(range(nbins)), density=False)
    cdf = imhist.cumsum() # cumulative distribution function (CDF) = cummulative histogram
    factor = 255 / cdf[-1]  # cdf[-1] = last element of the cummulative sum = total number of pixels)
    im2 = np.interp(im.flatten(), bins[:-1], factor*cdf)
    return im2.reshape(im.shape), cdf

def testHistEq(im):
    im2, cdf = histeq(im)
    return [im2, cdf]

def darkenImg(im,p=2):
    if len(im.shape) == 3:  # Проверка, является ли изображение цветным
        # Применение затемнения к каждому каналу
        return np.stack([(channel ** float(p)) / (255 ** (p - 1)) for channel in np.rollaxis(im, axis=-1)], axis=-1)
    else:  # Изображение в оттенках серого
        return (im ** float(p)) / (255 ** (p - 1))

def brightenImg(im,p=2):
    if len(im.shape) == 3:  # Проверка, является ли изображение цветным (3D массив)
    # Применение осветления к каждому каналу
        return np.stack([np.power(255.0 ** (p - 1) * channel, 1. / p) for channel in np.rollaxis(im, axis=-1)], axis=-1)
    else:  # Изображение в оттенках серого (2D массив)
        return np.power(255.0 ** (p - 1) * im, 1. / p)


def testDarkenImg(im):
    im2 = darkenImg(im,p=2) #  Is "p=2" different here than in the function definition? Can we remove "p=" here?
    return [im2]


def testBrightenImg(im):
    p=2
    im2=brightenImg(im,p)
    return [im2]

path_input = './1imgs-P1/'
path_output = './1imgs-out-P1/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.ppm")
else:
    files = [path_input + 'iglesia.pgm'] # iglesia,huesos

bAllTests = True
if bAllTests:
    tests = ['testHistEq', 'testBrightenImg', 'testDarkenImg']
else:
    tests = ['testHistEq']#['testBrightenImg']
nameTests = {'testHistEq': "Histogram equalization",
             'testBrightenImg': 'Brighten image',
             'testDarkenImg': 'Darken image'}
suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testDarkenImg': '_dk'}

bSaveResultImgs = True

def doTests():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile))  # from Image to array
        dirname, basename = os.path.dirname(imfile), os.path.basename(imfile)
        fname, fext = os.path.splitext(basename)  # Извлечение имени файла и расширения
        for test in tests:
            out = eval(test)(im)
            im2 = out[0]
            vpu.showImgsPlusHists(im, im2, title=nameTests[test])
            if len(out) > 1:
                vpu.showPlusInfo(out[1],"cumulative histogram" if test=="testHistEq" else None)
            if bSaveResultImgs:
               saveImg(im2, path_output, fname, suffixFiles[test] + fext)

if __name__== "__main__":
    doTests()

