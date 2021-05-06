# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import random
import cv2
import json
import math
import copy
import numpy as np
import os
from PIL import Image


class ImageAugmentor:  

    '''
    Load all the images or masks in a given folfer
    @param {folder} the folder name
    @param {is_img} if the foldeer contains image. True for images, false for masks
    @return A list of images that Opencv can operatee
    '''
    def load_images_from_folder(self, folder, is_img):
        images = []
        for filename in os.listdir(folder):
            if is_img: 
                img = cv2.imread(os.path.join(folder,filename))
                if img is not None: images.append(img)
            else: 
                _, masks = cv2.imreadmulti(os.path.join(folder,filename), [], cv2.IMREAD_ANYDEPTH)
                if len(masks) != 0: 
                    images.append(masks)
        return images

    '''
    Do a random augmentation of image and corresponding mask
    @param {image_route_list} the folder route for image
    @param {mask_route_list} the folder route for mask_route_list
    @return A tuple of images and masks, correspondingly. 
    '''
    def randomAugmentation(self, image_route_list, mask_route_list, apply_augmentation=True):

        pre_imgs = self.load_images_from_folder(image_route_list, True)
        pre_masks = self.load_images_from_folder(mask_route_list, False)

        if not apply_augmentation:
            return(pre_imgs, pre_masks)
        
        augmented_imgs = []
        augmented_masks = []
        

        for img, masks in zip(pre_imgs, pre_masks):
            # randomize the augmentation parameters
            params = self.randomize_params()

            # img = cv2.imread(img_path)
            # _, masks = cv2.imreadmulti(mask_path, [], cv2.IMREAD_ANYDEPTH)

            augmented_imgs.append(self.make_augmentation(img, params))

            for index in range(len(masks)):
                masks[index] = self.make_augmentation(masks[index], params, True)
            
            augmented_masks.append(masks)
            
        return (augmented_imgs, augmented_masks)
        

    '''
    Do a augmention on a list of points 2D
    @param {ponits} n*2 matrix of all the points
    @param {degree} degree of the rotation
    @param {zoom_factor} zoom factor
    @param {center} centeer point position
    @return a list of transformed points
    '''
    def points_augmentation(self, points, degree, center, zoom_factor):
        M = cv2.getRotationMatrix2D(center, degree, zoom_factor)
        transformed_points = np.asarray([(M[0][0] * x + M[0][1] * y + M[0][2],
                             M[1][0] * x + M[1][1] + M[1][2]) for (x, y) in points])
        return transformed_points

    

    # Randomize all the parameters
    def randomize_params(self):
        params = {}
        params["rotate"] = random.choice([20, 45, 60, 90, 180, 270])
        params["zoom"] = random.choice([1.05, 1.1, 1.15, 1.2])
        params["brightness"] = random.choice([0, -10, 10,   0,  0, 5])
        params["contrast"] = random.choice([0, 0, 0, -3, 5, 3])

        return params


    def make_augmentation(self, img, params, ismask=False, extra=None):

        # do the zoom
        # result = self.cv2_clipped_zoom(img, params["zoom"])

        # do the rotation
        result = self.cv2_rotate(img, params["rotate"], ismask=ismask, extra=extra)

        # do the bright and contrast
        if not ismask:
            result = self.cv2_brightness_contrast(result, params["brightness"], params["contrast"])

            # do the artifacts
            result = self.cv2_random_artifact(result)

        return result


    def rotate_point(self, x, y, cx, cy, degree):
        rad = (float(degree) / 180.0) * math.pi
        nx = (  (x - cx) * math.cos(rad) + (y - cy) * math.sin(rad) ) + cx
        ny = ( -(x - cx) * math.sin(rad) + (y - cy) * math.cos(rad) ) + cy
        return nx, ny


    def cv2_rotate(self, img, degree, ismask=False, extra=None):
        if isinstance(img, list):
            (h, w) = extra
        else:
            (h, w) = img.shape[:2]

        # calculate the center of the image
        center = (w / 2, h / 2)

        # do the rotation
        if ismask:
            nres = []
            for bbox in img:
                # convert to cartesian sets
                xmin = bbox['left']
                ymin = h - (bbox['top'] + bbox['height'])
                xmax = bbox['left'] + bbox['width']
                ymax = h - bbox['top']

                # rotate the points around the center
                n_xmin, n_ymin = self.rotate_point(xmin, ymin, center[0], center[1], degree)
                n_xmax, n_ymax = self.rotate_point(xmax, ymax, center[0], center[1], degree)

                # convert bbox points
                data = copy.deepcopy(bbox)
                data['left'] = math.floor(min(n_xmin, n_xmax))
                data['top'] = math.floor(h - max(n_ymin, n_ymax))
                data['width'] = math.ceil(max(n_xmin, n_xmax) - data['left'])
                data['height'] = math.ceil(data['top'] - (h - min(n_ymin, n_ymax)))
                nres.append(data)

            return nres

        M = cv2.getRotationMatrix2D(center, degree, 1)
        result = cv2.warpAffine(img, M, (h, w))

        return result


    def cv2_clipped_zoom(self, img, zoom_factor):
        height, width = img.shape[:2] # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

        ### Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1,x1,y2,x2])
        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(new_width, width)
        pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

        result = cv2.resize(cropped_img, (resize_width, resize_height))
        result = np.pad(result, pad_spec, mode='constant')
        assert result.shape[0] == height and result.shape[1] == width
        return result
    
    def cv2_brightness_contrast(self, input_img, brightness = 0, contrast = 0):

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def cv2_random_artifact(self, img):
        height, width = img.shape[:2]
        start_point = (random.randint(0, width), random.randint(0, height))
        end_point = (random.randint(0, width), random.randint(0, height))
        thickness = random.randint(1, 3)
        color = (0, 0, 0)
        result = cv2.line(img, start_point, end_point, color, thickness)
        return result



