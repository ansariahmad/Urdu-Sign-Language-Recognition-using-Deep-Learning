import os
import cv2

# train directory
train_dir = "dataset\captured_images\\train"

counter = 0
for dir in os.listdir(train_dir):
    gray_train_dir = os.path.join("dataset", "grayscale_images", "train", dir)
    if not os.path.exists(gray_train_dir):
        os.makedirs(gray_train_dir)
    for file in os.listdir(os.path.join(train_dir, dir)):
        img = cv2.imread(os.path.join(train_dir, dir, file))
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray_image, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(gray_train_dir, file), res)
    counter += 1
    print("{} folder completed".format(counter))
   
  
# test directory
test_dir = "dataset\captured_images\\test"

counter = 0
for dir in os.listdir(test_dir):
    gray_test_dir = os.path.join("dataset", "grayscale_images", "test", dir)
    if not os.path.exists(gray_test_dir):
        os.makedirs(gray_test_dir)
    for file in os.listdir(os.path.join(test_dir, dir)):
        img = cv2.imread(os.path.join(test_dir, dir, file))
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray_image, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(gray_test_dir, file), res)
    counter += 1
    print("{} folder completed".format(counter))
