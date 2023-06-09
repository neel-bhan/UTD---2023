import math
import pandas as pd
import matplotlib.pyplot as plt

column = ['col1', 'col2']
df = pd.read_csv('data.train', names=column)

list_x = df['col1'].tolist()
list_y = df['col2'].tolist()

print(list_x)
print(list_y)
def error(m, b, xL, yL):
    totalloss = 0
    num = len(xL)
    for i in range(num):
        x = xL[i]
        y = yL[i]
        totalloss += float((y - (m * x + b))**2)
    return totalloss / float(num)

def da(a, b, count):
    sum = 0
    for i in range(count):
        sum += 2 * (a * list_x[i] + b - list_y[i]) * list_x[i]
    return sum/count

def db(a, b,count):
    sum = 0
    for i in range(count):
        sum +=  2 * (a * list_x[i] + b - list_y[i])
    return sum/count

def ua(a, b, count):
    a = a - (0.000001 * da(a, b, count))
    return a

def ub(a, b, count):
    b = b - (0.000001 * db(a, b, count))
    return b

a = 3
b = 2

for i in range(1000):
    a = ua(a, b, len(list_x))
    b = ub(a, b, len(list_x))


plt.scatter(df['col1'], df['col2'])
for i in range(-1000, 1000, 50):
    plt.scatter(i, (a * i + b))
plt.show()
print(error(a, b,list_x, list_y))


