from PIL import Image


picture = Image.open('C:\\Users\\david\\Resources\\Images\\arrow_190cm_1679703098.5790615.jpg')
 

r,g,b = picture.getpixel((412,113)) # gets RBG from those specfic pixels

print(("R: {0}, G: {1} B: {2}").format(r,g,b))

new_color = (500,20,300)


width = list(range(960))
height = list(range(720))


arrow_array_orange = [[],[]]
arrow_array_blue = [[],[]]
not_arrow_array = [[], []]

def arrow_color(array_o, array_b):
    for x in width:  # scans every pixel looking for r values less than 4.
        for y in height:
            r,g,b = picture.getpixel((x,y))
            if r > 90 and r < 150 and g > 40 and g < 75 and b >=0 and b < 25:
                array_o[0].append(x) # puts those values (x_pixel and y_pixel) which are less than 4 into an array
                array_o[1].append(y)
            if r > 6 and r < 30 and g > 15 and g < 35 and b > 54 and b < 100:
                array_b[0].append(x)
                array_b[1].append(y)

    for i in range(len(array_o[0])): # changes the color of the pixels selected.
        x = array_o[0][i]
        y = array_o[1][i]
        
        #print(x,y)
        picture.putpixel((x,y), new_color)
        
    
    for i in range(len(array_b[0])):
        x_2 = array_b[0][i]
        y_2 = array_b[1][i]
        picture.putpixel((x_2,y_2), (30,144,255))
    picture.save('C:\\Users\\david\\Resources\\Images\\new_arrow.jpg')

            
            
def background_gray(array):
    new_picture = Image.open('C:\\Users\\david\\Resources\\Images\\new_arrow.jpg')
    for x in width:  # scans every pixel looking for r values less than 4.
        for y in height:
            r,g,b = new_picture.getpixel((x,y))
            #print(("R: {0}, G: {1} B: {2}").format(r,g,b))
            if r != 50 and r!= 144 and g != 20 and g!=144 and b!= 300 and b!=255:
                array[0].append(x) # puts those values (x_pixel and y_pixel) which are less than 4 into an array
                array[1].append(y)
    
    for i in range(len(array[0])): # changes the color of the pixels selected.
        x = array[0][i]
        y = array[1][i]
        #print(x,y)
        new_picture.putpixel((x,y), (1,1,1))
    new_picture.save('C:\\Users\\david\\Resources\\Images\\new_arrow.jpg')
            
arrow_color(arrow_array_orange, arrow_array_blue)
background_gray(not_arrow_array)





