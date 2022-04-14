import torch
import kornia

import cv2
import numpy as np
import matplotlib.pyplot as plt


#the [x, y] for each right-click event will be stored here
right_clicks = list()

#this function will be called whenever the mouse is left-clicked
def mouse_callback(event, x, y, flags, params):

    if event == 1:

        global right_clicks, img

        #store the coordinates of the right-click event
        right_clicks.append([x, y])
        # draw the blue circle
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('image', img)
        # if we have clicked 4 times
        if len(right_clicks) == 4:
            cv2.destroyAllWindows()
            

# This is a function responsable by image
def get_image(path_image: str):

    # Load image
    img = cv2.imread(path_image,1)

    # Get width and height from image and resize
    scale_width = 640 / img.shape[1]
    scale_height = 480 / img.shape[0]
    scale = min(scale_width, scale_height)
    
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    
    # create window for mouse_callback points
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)

    #set mouse callback function for window
    cv2.setMouseCallback('image', mouse_callback)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

img = get_image(r"data\batman.jpg")


# Load image with Kornia
image = kornia.image_to_tensor(img)
image = torch.unsqueeze(image.float(), dim=0)  # BxCxHxW

# the source points are the region to crop corners
points_src = torch.FloatTensor([right_clicks])

# the destination points are the image vertexes
h, w = 480, 640  # destination size
points_dst = torch.FloatTensor([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]])

# compute perspective transform
M = kornia.geometry.transform.get_perspective_transform(points_src, points_dst)
# warp the original image by the found transform
img_warp = kornia.geometry.transform.warp_perspective(image, M, dsize=(h, w))


# convert back to numpy
image_warp = kornia.tensor_to_image(img_warp.byte()[0])

# draw points into original image
for i in range(4):
    center = tuple(points_src[0, i].long().numpy())
    image = cv2.circle(img.copy(), center, 5, (0, 255, 0), -1)


# Visualize image with matplotlib
fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('Image Original')
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

axs[1].axis('off')
axs[1].set_title('Image Perspective')
axs[1].imshow(cv2.cvtColor(image_warp, cv2.COLOR_BGR2RGB))
plt.show()
