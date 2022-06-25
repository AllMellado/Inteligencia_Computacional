import os
import pathlib

filepath = "a"

with open(filepath, "r") as file:


bias = 0.02
x0 = -1
passo = 0.01

bestMSE = 500
MSE = 500
error = 1
it = 1

while MSE != 0 and it <= 200:
    error = 0
    MSE = 0
    for