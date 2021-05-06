from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import random
import cv2
from image_augmentor import ImageAugmentor
import numpy as np
import os
import math



class ImageChunker:  

    '''
    split a image into small pieces
    @param {img} an Image object loaded by opencv
    @param {size} size of the small image that you want
    @return A list of small pieces of image
    '''
    def chunk_image(self, img, size):
        chunks = []
        augmentor = ImageAugmentor()
        params = augmentor.randomize_params()
        height, width = img.shape[:2]

        x_num = math.floor(width / size)
        y_num = math.floor(height / size)

        for y in range(y_num):
            for x in range(x_num):
                crop = img[y * size:y * size + size, x * size:x * size + size]
                crop = augmentor.make_augmentation(crop, params)
                chunks.append(crop)
        return chunks

    '''
    split a image into small pieces and put them into batches
    @param {img} an Image object loaded by opencv
    @param {size} size of the small image that you want
    @param {num_of_iteration} number of iteration
    @return Batch
    '''
    def split_to_batch(self, img, size, num_of_iteration):
        chunks = chunk_image(img, size)
        batch_size = math.floor(len(chunks) / num_of_iteration)
        batches = []
        for i in range(num_of_iteration):
            batches[i] = chunks[i * batch_size : i * batch_size + batch_size]
        return batches
        


  