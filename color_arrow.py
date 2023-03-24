from PIL import Image


picture = Image.open('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows\\arrow_178cm_1679114955.2507687.jpg')


r,g,b = picture.getpixel((629,19)) # gets RBG from those specfic pixels

print(("R: {0}, G: {1} B: {2}").format(r,g,b))

new_color = (500,20,300)


width = list(range(960))
height = list(range(720))


my_array = [[],[]]

for x in width:  # scans every pixel looking for r values less than 4.
    for y in height:
        r,g,b = picture.getpixel((x,y))
        if r > 1 and r < 5 and g > 18 and g <22 and b > 50 and b < 57:
            my_array[0].append(x) # puts those values (x_pixel and y_pixel) which are less than 4 into an array
            my_array[1].append(y)



for i in range(len(my_array[0])): # changes the color of the pixels selected.
    x = my_array[0][i]
    y = my_array[1][i]
    #print(x,y)
    picture.putpixel((x,y), new_color)

picture.save('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows\\arrow_178cm_1679114955.2507687.jpg')

