import numpy as np
import codecademylib3
from math import sin, cos, log, pi
import matplotlib.pyplot as plt

def limit_derivative(f, x, h):
  """
  f: function to be differentiated 
  x: the point at which to differentiate f 
  h: distance between the points to be evaluated
  """
  # compute the derivative at x with limit definition
  return (f(x + h) - f(x)) / h

# f1(x) = sin(x)
def f1(x):
    return sin(x)

# f2(x) = x^4
def f2(x):
    return pow(x, 4)

# f3(x) = x^2*log(x)
def f3(x):
    return pow(x, 2) * log(x)

# Calculate derivatives here

# Graph the true derivative
x_vals = np.linspace(1, 10, 200)
y_vals = [cos(val) for val in x_vals]
plt.figure(1)
plt.plot(x_vals, y_vals, label="True Derivative", linewidth=4)

# plot different approximated derivatives of f using limit definition of derivative
def plot_approx_deriv(f):
  x_vals = np.linspace(1, 10, 200)
  h_vals = [10, 1, .25, .01]

  for h in h_vals:
      derivative_values = []
      for x in x_vals:
          derivative_values.append(limit_derivative(f, x, h))
      plt.plot(x_vals, derivative_values, linestyle='--', label="h=" + str(h) )
  plt.legend()
  plt.title("Convergence to Derivative by varying h")
  plt.show()

# Calculate the derivative of f3 at x=1 for different h values
h_values = [2, 0.1, 0.00001]
x = 1

for h in h_values:
    derivative = limit_derivative(f3, x, h)
    print(f"Derivative of f3 at x={x} with h={h}: {derivative}")

# Calculate the derivative of f1 at x=pi/2 for different h values
h_values = [2, 0.1, 0.00001]
x_f1 = pi / 2

for h in h_values:
    derivative_f1 = limit_derivative(f1, x_f1, h)
    print(f"Derivative of f1 at x={x_f1} with h={h}: {derivative_f1}")

# Calculate the derivative of f2 at x=2 for different h values
x_f2 = 2

for h in h_values:
    derivative_f2 = limit_derivative(f2, x_f2, h)
    print(f"Derivative of f2 at x={x_f2} with h={h}: {derivative_f2}")

plot_approx_deriv(f3)

y_vals = [cos(val) for val in x_vals]

# Comment out or delete this line if it exists
# plot_approx_deriv(f1)

# Add this line to plot the derivative approximations for f2
plot_approx_deriv(f2)


