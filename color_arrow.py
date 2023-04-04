from PIL import Image
import time
import numpy as np
import os
import cv2

arrow_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\unfiltered_arrow_pics'

new_color = (500,20,300)


width = list(range(960))
height = list(range(720))

def arrow_color(picture):
    picture = Image.open(picture)
    #picture = Image.fromarray(np.uint8(picture))  # convert NumPy array to PIL image
    array_o = [[],[]]
    array_b = [[],[]]
    array = [[], []]

    for x in width:  # scans every pixel looking for r values less than 4.
        for y in height:
            r,g,b = picture.getpixel((x,y))
            if r > 50 and r < 260 and g > 16 and g < 160 and b >=0 and b < 25:
                array_o[0].append(x) # puts those values (x_pixel and y_pixel) which are less than 4 into an array
                array_o[1].append(y)
            if r > 0 and r < 75 and g > 1 and g < 105 and b > 35 and b < 195:
                array_b[0].append(x)
                array_b[1].append(y)

    for i in range(len(array_o[0])): # changes the color of the pixels selected.
        x = array_o[0][i]
        y = array_o[1][i]
        
        #print(x,y)
        picture.putpixel((x,y), (255,97,3))
        
    
    for i in range(len(array_b[0])):
        x_2 = array_b[0][i]
        y_2 = array_b[1][i]
        picture.putpixel((x_2,y_2), (0,0,255))

    for x in width:  
        for y in height:
            r,g,b = picture.getpixel((x,y))
            #print(("R: {0}, G: {1} B: {2}").format(r,g,b))
            if r != 255 and r!= 0 and g != 97 and g!=0 and b!= 3 and b!=255:
                array[0].append(x) # puts those values (x_pixel and y_pixel) which are less than 4 into an array
                array[1].append(y)
    
    for i in range(len(array[0])): # changes the color of the pixels selected.
        x = array[0][i]
        y = array[1][i]
        #print(x,y)
        picture.putpixel((x,y), (112,128,144))
    picture.save(f'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows\\85.cm_{time.time()}.jpg')

    #return picture
    #return np.array(picture)



img = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\unfiltered_arrow_pics\\85.cm_1680313048.2506518.jpg'

test_pixel = Image.open(img)

r,g,b = test_pixel.getpixel((537,225)) # gets RBG from those specfic pixels

print(("R: {0}, G: {1} B: {2}").format(r,g,b))

arrow_color(img)






