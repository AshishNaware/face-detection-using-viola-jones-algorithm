# importing libraries
import cv2
import os
import numpy as np

# function going through every directory in test images
def build_train_data(faces, non_faces):
   faces_list = []
   non_faces_list = []
   iterable = [faces, non_faces]
   for item in iterable:
      for root, dirs, files in os.walk(item, topdown=True):
         # don't need roots and directories because we already have set condition of testimgs folder and only care about files
         for name in files:
            # making sure we are reading only .jpg files
            if name.endswith('.jpg'):
               # building path of image
               img_path = os.path.join(root, name)
               # reading, resizing, filtering, and showing image at that path
               img = cv2.imread(img_path, flags=0)
               img = cv2.resize(img, (24, 24))
               if item == faces:
                  faces_list.append(img)
               else:
                  non_faces_list.append(img)
   return faces_list, non_faces_list

# gets integral images
def get_integral_imgs(images):
   integral_imgs = []
   for image in images:
      img_shape = image.shape
      integral_img = np.empty(img_shape, dtype=int)
      for i in range(img_shape[0]):
         for j in range(img_shape[1]):
            if (i, j) == (0, 0):
               integral_img[i, j] = image[i, j]
            elif i == 0:
               integral_img[i, j] = integral_img[i, j - 1] + image[i, j]
            elif j == 0:
               integral_img[i, j] = integral_img[i - 1, j] + image[i, j]
            else:
               integral_img[i, j] = image[i, j] + integral_img[i - 1, j] + integral_img[i, j - 1] - integral_img[
                  i - 1, j - 1]

      integral_imgs.append(integral_img)
      return np.asarray(integral_imgs)


def main():
   faces_list, non_faces_list = build_train_data('./testimgs/test1', './testimgs/test2')
   faces = get_integral_imgs(faces_list)
   non_faces = get_integral_imgs(non_faces_list)


if __name__ == "__main__":
    main()
