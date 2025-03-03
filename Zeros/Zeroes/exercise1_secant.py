"""Same exercise of lagrangian point done with Newton method"""


# Parameters
G = 6.674e-11
M = 5.974e+24
m = 7.348e+22
R = 3.844e+8
omega = 2.662e-6

def newton_secant(func, x0, deriv=None, x1=None, tol=1e-4):
    # Finds a solution of func(x)=0 using Newton's method
    # deriv is an input function which computes df/dx
    # x0 is the starting point of the search
    # tol is the tolerance, set by default at 1e-4

    demomode = True

    acc = tol + 1.

    # If deriv is not passed, use secant
    if (deriv == None):

        if demomode:
            print('Using secant method')

        # If x1 is not passed for secant method,
        # choose x1 = x0 + 1 by default
        if (x1 == None):
            x1 = x0 + 1.

        while (acc > tol):
            fprime = (func(x1) - func(x0)) / (x1 - x0)
            delta = func(x1) / fprime
            x2 = x1 - delta
            if demomode:
                print('root = ', x2, 'acc = ', abs(delta))
            acc = abs(x2 - x1)
            x0, x1 = x1, x2

        x = x2

    # If deriv is passed, use Newton
    else:

        if demomode:
            print('Using Newton method')

        x = x0

        while (acc > tol):
            delta = func(x) / deriv(x)
            x1 = x - delta
            if demomode:
                print('root = ', x1, 'acc = ', abs(delta))
            acc = abs(x1 - x)
            x = x1

    return x

# Function
f = lambda r: (G*M)/(r**2.) - (G*m)/(R-r)**2. - omega**2*r
# Derivative
deriv = lambda r: -(2.*G*M)/(r**3.) - (2.*G*m)/(R-r)**3. - omega**2.

# Solving for position of L1, using Newton's method
start = R/100.

# Uses secant method with default x1
#L1 = newton_secant(f,start,tol=1e-6)
# Uses secant method with x1 passed by user
L1 = newton_secant(f,start,x1=start+R/10.,tol=1e-6)
# Uses Newton's method
#L1 = newton_secant(f,start,deriv,tol=1e-6)

print(' ')
print('The root in units of R is: x/R = ', L1/R)