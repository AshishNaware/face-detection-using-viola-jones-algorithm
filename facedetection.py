# importing libraries
import cv2
import os

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




def main():
   faces_list, non_faces_list = build_train_data('./testimgs/test1', './testimgs/test2')
   print(
      len(faces_list) + len(non_faces_list)
   )


if __name__ == "__main__":
    main()
