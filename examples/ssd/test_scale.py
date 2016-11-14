from scale import scale_box

# new_dim = (640, 360)
# current_dim = (500, 500)
new_dim = (10, 10)
current_dim = (5, 5)
# xfact = float(current_dim[0])/float(new_dim[0])
# yfact = float(current_dim[1])/float(new_dim[1])
xfact = float(new_dim[0])/float(current_dim[0])
yfact = float(new_dim[1])/float(current_dim[1])
xmin = 1
xmax = 2
ymin = 1
ymax = 2

print 'x fact {:06.2f}, y fact {:06.2f}'.format(xfact, yfact)
print '{:06.2f}, {:06.2f}, {:06.2f}, {:06.2f},'.format(xmin, ymin, xmax, ymax)
new_box = scale_box(xmin, ymin, xmax, ymax, xfact, yfact)
print '{:06.2f}, {:06.2f}, {:06.2f}, {:06.2f},'.format(new_box[0], new_box[1], new_box[2], new_box[3]) 


