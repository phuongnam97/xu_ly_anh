import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'image/'
fileName = 'bo-dan-ga-3-compressed.jpg'
img = cv2.imread(path + fileName)
# chuyen thanh anh xam
gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# can bang histogram
im_equal = cv2.equalizeHist(gray_im)

# gray_correct = np.array(255 * (gray_im / 255) ** 1.2, dtype='uint8')

thresh = cv2.adaptiveThreshold(im_equal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 20)

# thresh = cv2.bitwise_not(thresh)

kernel = np.ones((15, 15), np.uint8)
# Phép co
img_erode = cv2.erode(thresh, kernel, iterations=1)
# Phép giãn
img_dilation = cv2.dilate(img_erode, kernel, iterations=1)

ret, labels = cv2.connectedComponents(img_dilation)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

print(ret)
listValue = []
count1 = 0
count2 = 0
for label in labels:
    count1 += 1

for label in labels[0]:
    count2 += 1

# lấy ds label
# for x in range(count2):
#     for y in range(count1):
#         if labels[y][x] != 0:
#             # print("x: " + str(x) + "  y : " + str(y) + "value " + str(labels[y][x]))
#             if labels[y][x] not in listValue:
#                 listValue.append(labels[y][x])
#
# print(listValue)
#
# # lấy obj so với mask
# for value in listValue:
#     maxX = 0
#     maxY = 0
#     minX = 1000000
#     minY = 1000000
#
#     for x in range(count2):
#         for y in range(count1):
#             if labels[y][x] == value:
#                 if x > maxX:
#                     maxX = x
#                 if y > maxY:
#                     maxY = y
#                 if x < minX:
#                     minX = x
#                 if y < minY:
#                     minY = y
#     print("===============")
#     print(maxX)
#     print(maxY)
#     print(minX)
#     print(minY)
#     print(maxX - minX)
#     print(maxY - minY)

plt.subplot(222)
plt.title('Objects counted:' + str(ret - 1))
plt.imshow(labeled_img)
# plt.imshow(gray_im)
print('objects number is:', ret - 1)

plt.show()
