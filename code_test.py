from image_chunker import ImageChunker
import cv2

# k = [1,2,3,4,5]
# print(k[1:2])

# augmentor = ImageAugmentor()
# both = augmentor.randomAugmentation(image_route_list="imgs", mask_route_list="masks")
# print(both[0][0].shape)
# print(len(both[1][0]))

chunker = ImageChunker()
img = cv2.imread("08_00815.png")
chunks = chunker.chunk_image(img, 256)
cv2.imwrite("starry_night.png", chunks[2])

