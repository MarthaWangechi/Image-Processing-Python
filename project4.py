import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

# this makes image look better on a macbook pro
def imageshow(img, dpi=200):
    if dpi > 0:
        F = plt.gcf()
        F.set_dpi(dpi)
    plt.imshow(img)


def rgb_ints_example():
    '''should produce red,purple,green squares
    on the diagonal, over a black background'''
    # RGB indexes
    red,green,blue = range(3)#0,1,2
    # img array 
    # all zeros = black pixels
    # shape: (150 rows, 150 cols, 3 colors)
    img = np.zeros((150,150,3), dtype=np.uint8)
    for x in range(50):
        for y in range(50):
            # red pixels
            img[x,y,red] = 255
            # purple pixels
            # set all 3 color components
            img[x+50, y+50,:] = (128, 0, 128)
            # green pixels
            img[x+100,y+100,green] = 255
    return img
"""
plt.imshow(rgb_ints_example())
pattern = plt.imread('pattern.png ')
imageshow(pattern)
"""
def onechannel(pattern, rgb):
    image = np.zeros(pattern.shape, dtype=pattern.dtype)
    for x in range(0,pattern.shape[0]):
        for y in range(0,pattern.shape[1]):
            image[x,y,rgb] = pattern[x,y,rgb]            
    return image
        
    
def permutecolorchannels(img,perm):
    image = np.zeros(img.shape, dtype=img.dtype)
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            for i in range(0,img.shape[2]):
                image[x,y,perm[i]] = img[x,y,i]    
    return image
    
def decrypt(image,key):
    new_image = np.zeros(image.shape, dtype=image.dtype)
    for x in range(0,image.shape[0]):
        for y in range(0,image.shape[1]):
            for c in range(3):                
                new_image[x,y,c] = image[x,y,c]^key[y]
    return new_image
    
if __name__ == "__main__":
    pattern = plt.imread('pattern.png')
    imageshow(pattern)    
    plt.pause(0.001)
    
    plt.imshow(onechannel(pattern, 0))
    plt.pause(0.001)
    
    plt.imshow(onechannel(pattern, 1))
    plt.pause(0.001)
    
    plt.imshow(onechannel(pattern, 2))
    plt.pause(0.001)
    
    plt.imshow(permutecolorchannels(pattern, [2,0,1]))
    plt.pause(0.001)
    
    permcolors = plt.imread('permcolors.jpeg')
    imageshow(permcolors)
    #displaying image with correct colors
    plt.imshow(permutecolorchannels(permcolors, [1,2,0]))
    plt.pause(0.001)
    
    secret = plt.imread('secret.bmp')
    plt.imshow(secret)
    plt.pause(0.001)
    
    key = np.load('key.npy')
    #decrypting the secret image
    plt.imshow(decrypt(secret,key))
