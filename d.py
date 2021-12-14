import math

a = [28, 24, 22, 20, 18]

def conv2(x, k):
    return math.floor((x + 2 * 0 - k)/1 + 1)

def maxPool(x, k):
    return math.floor((x + 2 * 0 - k)/k + 1)

for x in a:
    a1 = conv2(x, 4)
    a2 = maxPool(a1, 2)
    a3 = conv2(a2, 5)
    a4 = maxPool(a3, 3)
    print(x, a1, a2, a3, a4)
