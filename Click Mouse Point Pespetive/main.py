import torch
import kornia

import cv2
import numpy as np
import matplotlib.pyplot as plt


right_clicks = list()

def mouse_callback(event, x, y, flags, params):

    if event == 1:

        global right_clicks, img

        right_clicks.append([x, y])
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('image', img)
        if len(right_clicks) == 4:
            cv2.destroyAllWindows()
            

def get_image(path_image: str):

    img = cv2.imread(path_image,1)

    scale_width = 640 / img.shape[1]
    scale_height = 480 / img.shape[0]
    scale = min(scale_width, scale_height)
    
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)

    cv2.setMouseCallback('image', mouse_callback)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

img = get_image(r"data\batman.jpg")


image = kornia.image_to_tensor(img)
image = torch.unsqueeze(image.float(), dim=0)  # BxCxHxW

points_src = torch.FloatTensor([right_clicks])

h, w = 480, 640  # destination size
points_dst = torch.FloatTensor([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]])

M = kornia.geometry.transform.get_perspective_transform(points_src, points_dst)
# warp the original image by the found transform
img_warp = kornia.geometry.transform.warp_perspective(image, M, dsize=(h, w))


image_warp = kornia.tensor_to_image(img_warp.byte()[0])

for i in range(4):
    center = tuple(points_src[0, i].long().numpy())
    image = cv2.circle(img.copy(), center, 5, (0, 255, 0), -1)


fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('Image Original')
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

axs[1].axis('off')
axs[1].set_title('Image Perspective')
axs[1].imshow(cv2.cvtColor(image_warp, cv2.COLOR_BGR2RGB))
plt.show()
