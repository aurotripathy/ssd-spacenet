from scale import scale_box

new_dim = (640, 360)
current_dim = (500, 500)
xfact = float(current_dim[0])/float(new_dim[0])
yfact = float(current_dim[1])/float(new_dim[1])
xmin = 5
xmax = 10
ymin = 5
ymax = 10

print 'x fact {:06.2f}, y fact {:06.2f}'.format(xfact, yfact)
print '{:06.2f}, {:06.2f}, {:06.2f}, {:06.2f},'.format(xmin, ymin, xmax, ymax)
new_box = scale_box(xmin, ymin, xmax, ymax, xfact, yfact)
print '{:06.2f}, {:06.2f}, {:06.2f}, {:06.2f},'.format(new_box[0], new_box[1], new_box[2], new_box[3]) 


