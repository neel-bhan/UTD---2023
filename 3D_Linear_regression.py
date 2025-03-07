import numpy as np
import matplotlib.pyplot as plt
xlist= []
ylist= []

#data import
data = np.recfromcsv('Sample_Data2.csv',)
data = np.array(data)
x, y, z= [], [], []
for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1])
    z.append(data[i][2])

ax = plt.axes(projection='3d')

ax.scatter(x, y, z, color = 'black')

del data

xmin, xmax = min(x), max(x)
ymin, ymax = min(y), max(y)
zmin, zmax = min(z), max(z)
#normalise
for i in range(len(x)):
    x[i] = (x[i] - xmin) / (xmax - xmin)
    y[i] = (y[i] - ymin) / (ymax - ymin)
    z[i] = (z[i] - zmin) / (zmax - zmin)




#powers
degrees = [j for j in range(5)]
degrees_inversed = [j for j in range(4, -1, -1)]

print(degrees)
print(degrees_inversed)
def equation(j):
    equation = 0
    for i in range(len(degrees)):
        var = weights[i] * x[j] ** degrees_inversed[i] * y[j] ** degrees[i]
        equation += var
    equation += bias

    return equation


def regression(deg_range, l):
# Initialize previous_loss with infinity

    for i in range(1000):
        # Calculate derivatives of mean squared error
        da = 2 * (equation(i) - z[i]) * x[i] ** degrees_inversed[0] * y[i] ** degrees[0]
        db = 2 * (equation(i) - z[i]) * x[i] ** degrees_inversed[1] * y[i] ** degrees[1]
        dc = 2 * (equation(i) - z[i]) * x[i] ** degrees_inversed[2] * y[i] ** degrees[2]
        dd = 2 * (equation(i) - z[i]) * x[i] ** degrees_inversed[3] * y[i] ** degrees[3]
        de = 2 * (equation(i) - z[i]) * x[i] ** degrees_inversed[4] * y[i] ** degrees[4]
        dbias = 2 * (equation(i) - z[i])

        # Update weights
        weights[0] -= da * l
        weights[1] -= db * l
        weights[2] -= dc * l
        weights[3] -= dd * l
        weights[4] -= de * l
        weights[5] -= dbias * l

        # Update loss

    return weights

# Rest of the code...


a, b, c, d, e, bias =0.3545508849504344 ,1.3668565168706093,-1.551607225979021,1.4082803366778862,-0.21410016435472914 ,0.1451482610568061


weights = [a, b, c, d, e, bias]
#regression(5)

for i in range(10000):
    a, b, c, d, e, bias = regression(4 , 0.001)

b = b + 0.25
print(a,b,c,d,e, bias)
    #print(a, b, c, d)

print(str(a) + ' ,' + str(b) + ',' + str(c) + ',' + str(d) + ',' + str(e) + ' ,' + str(bias))

xvals, yvals, zvals = [], [], []
for i in range(1000):
    xval = x[i]
    yval = y[i]
    zval = a * xval**4 + \
           b * xval**3 * yval + \
           c  * xval** 2 * yval**2 + \
           d * xval * yval**3 + \
           e * yval**4+ bias


    xvals.append(xval)
    yvals.append(yval)
    zvals.append(zval)

for i in range(1000):
    xvals[i] = xvals[i] * (xmax - xmin) + xmin
    yvals[i] = yvals[i] * (ymax - ymin) + ymin
    zvals[i] = zvals[i] * (zmax - zmin) + zmin

ax.scatter(xvals, yvals, zvals, color = 'red')


plt.show()



















