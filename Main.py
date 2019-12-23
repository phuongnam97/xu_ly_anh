import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

original = cv.imread('image/oranges-organic-farmdrop18-March2010641.jpg')
thresh_type_inv = False

# original = cv.imread('image/danga.jpg')
# thresh_type_inv = True

f, axes = plt.subplots(2, 3)

thresh_value = 235
dil_ero_value_x = 15
dil_ero_value_y = 15
ksize_2 = 7


# thresh_type_inv = Truealse

def show():
    axes[0][0].set_title('Origin')
    axes[0][0].imshow(original)

    # Convert image in grayscale
    img = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    axes[0][1].set_title('Grayscale image')
    axes[0][1].imshow(img, cmap="gray", vmin=0, vmax=255)
    cv.imwrite("GrayScale.jpg", img)

    # Contrast adjusting with histogramm equalization
    img = cv.equalizeHist(img)
    axes[0][2].set_title('Histogram equalization')
    axes[0][2].imshow(img, cmap="gray", vmin=0, vmax=255)
    cv.imwrite("HistEqual.jpg", img)

    # Local adaptative threshold
    if thresh_type_inv:
        thresh_type = cv.THRESH_BINARY_INV
    else:
        thresh_type = cv.THRESH_BINARY
    t_val, img = cv.threshold(img, thresh_value, 255, thresh_type)
    cv.imwrite("Thresh.jpg", img)
    # img = cv.bitwise_not(img)

    # img = cv.medianBlur(img, ksize=ksize_1)
    axes[1][0].set_title(
        'thresh val =' + str(t_val)
    )
    axes[1][0].imshow(img, cmap="gray", vmin=0, vmax=255)

    # Dilatation et erosion
    kernel = np.ones((dil_ero_value_x, dil_ero_value_y), np.uint8)
    img = cv.dilate(img, kernel, iterations=1)
    cv.imwrite("dilate.jpg", img)
    img = cv.erode(img, kernel, iterations=1)
    cv.imwrite("erode.jpg", img)
    # clean all noise after dilatation and erosion
    img = cv.medianBlur(img, ksize=ksize_2)
    cv.imwrite("Median_blur.jpg", img)
    axes[1][1].set_title(
        'D&E (nm,.): ' + str(dil_ero_value_x) + '-' + str(dil_ero_value_y) + '\nksize(cv): ' + str(ksize_2))
    axes[1][1].imshow(img, cmap="gray", vmin=0, vmax=255)

    # Labeling
    ret, labels = cv.connectedComponents(img)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    img = cv.merge([label_hue, blank_ch, blank_ch])
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    img[label_hue == 0] = 0
    axes[1][2].imshow(img)
    axes[1][2].set_title('Objects counted:' + str(ret - 1))
    cv.imwrite("label.jpg", img)


show()
plt.show()
