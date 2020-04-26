import numpy as np

def foo(arr):
   arr = arr - 3

def bar(arr):
   arr -= 3

a = np.array([3, 4, 5])
foo(a)
print(a) # prints [3, 4, 5]

bar(a)
print(a) # prints [0, 1, 2]