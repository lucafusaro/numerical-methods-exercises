"""Blackbody law:  I(lambda) = 2 pi h_bar c^2 lambda^-5 / ( e^(h_bar c / lambda K_B T) - 1)
Differentiate and show that the maximum of emitted radiation is the solution of the equation  5 e^-x + x − 5=0
 , where  𝑥 = ℎ𝑐/𝜆𝑘𝐵𝑇
Therefore the peak wavelenght is  𝜆=𝑏/𝑇, with  𝑏=ℎ𝑐/𝑘𝐵𝑥
Solve the equation above for x with binary search (bisection), with accuracy 1e-6.
Estimate the temperature of the sun, knowing that the peak wavelength of emitted radiation is  𝜆=502 nm."""

import numpy as np
import sys
import matplotlib.pyplot as plt

def bisection(f, x1, x2, accuracy):
    while abs(x1 - x2) > accuracy:
        if f(x1) > 0 > f(x2) or f(x1) < 0 < f(x2):
            #mid point
            x = 0.5 * (x1 + x2)
            if f(x) * f(x1) > 0: #same sign
                x1 = x
            else:
                x2 = x
        else:
            sys.exit('f(x) does not have opposite signs at the boundaries')
        x = 0.5 * (x1 + x2)
    return x

def f(x):
    func = 5 * np.exp(-x) + x - 5
    return func

x = np.arange(-3, 15, 0.1)
plt.axhline(y=0, linestyle="--", color="red")
plt.plot(x, f(x))
plt.xlabel("x")
plt.ylabel("y = f(x)")
plt.show()

accuracy = 1e-6
x1 = 2
x2 = 10
root = bisection(f, x1, x2, accuracy)
print("La prima radice è x=", root)
print("Ad occhio vediamo subito guardando f(x) che c'è un'altra radice per x = 0")
#??? radice ad occhio perchè sennò ritorna valori molto piccoli ma non zero, si potrebbe altrimenti
#inserire una consizione nella funzione bisection per cui se x < 10^-5 per esempio restituisce zero