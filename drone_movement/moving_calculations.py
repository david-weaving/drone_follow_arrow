import math

def find_direction(fit_x, fit_y, fit_h):
    fit_x # fit_x is the predicted x_pixel value
    fit_y # fit_y is the predicted y_pixel value
    fit_h # fit_h is the predicted distance value

    L = 92.3 # distance from the wall in cm
    x = (fit_x-480)*(98/960)
    y = (577-fit_y)*(80/720) # 577 - fit_y because the top pixel is zero and the bottom is 720


    z_prime = (fit_h/math.sqrt((x*x)/(L*L) + (y*y)/(L*L) + 1)) # real z distance (moves back and forth)

    x_prime = (z_prime*(x/L)) # real x distance (moves left and right)

    y_prime = (z_prime*(y/L)) # real y distance (moves up and down)

    return z_prime, x_prime, y_prime
