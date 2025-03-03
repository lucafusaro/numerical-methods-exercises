"""Write your own function for trapezoidal integration.
The function should double the number of points in the sampled interval
  until a specifified level of accuracy is reached"""

def trap(func, a, b, tol):
    # Initial number of sampled points
    N = 10

    # 1/2*f(a) + 1/2*f(b)
    s = 0.5 * func(a) + 0.5 * func(b)

    # Starting with N points
    h = (b - a) / N
    s1 = 0.
    for k in range(1, N):
        s1 += func(a + k * h)
    # Trapezoidal rule with N points in [a,b]
    trap1N = h * (s + s1)

    acc = tol + 1.

    while (acc > tol):
        # Now doubling the number of points and updating h
        N = 2 * N
        h = (b - a) / N
        # Computing \sum f(x), only in the new sampled points
        s2 = 0.
        for k in range(1, N, 2):
            s2 += func(a + k * h)
        # Trapezoidal rule with 2N points
        trap2N = h * (s + s1 + s2)

        acc = ((1. / 3.) * abs(trap2N - trap1N)) / abs(trap2N)

        # Update and loop
        s1, trap1N = s1 + s2, trap2N

        # Exit loop if N > 1e6
        if (N > 1e6):
            print(' ')
            print('The required accuracy could not be reached with N=1e6 points.')
            print('Stopping here, N =', N, '; acc =', acc)
            break

    return trap2N