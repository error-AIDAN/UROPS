
import numpy as np
import matplotlib.pyplot as plt

# Constants
T = np.pi
K = 4
Δ = 1/4

# Define the time points for different scenarios
time_points = np.array([np.pi*k/K for k in range(K)])

# Define the function to calculate Lx(0)
def Lx(x0, p0, t):
    return x0 * np.cos(t) + p0 * np.sin(t)

# Define the function to check if x0 is within ±Δ
def check_delta(x0):
    return np.abs(x0) <= Δ/2
# Calculate S(j,Δ) for the grid
def S(x0, p0):
    sum_prob = 0
    for k in range(K):
        sum_prob += (-1)**k * check_delta(Lx(x0, p0, np.pi*k/K))
    return sum_prob / K

# Create the grid
x0_range = np.linspace(-5, 5, 600)
p0_range = np.linspace(-5, 5, 600)
x0, p0 = np.meshgrid(x0_range, p0_range)

# Circular mask
radius = 5  # Define the radius of the circle for the mask
circle_mask = x0**2 + p0**2 <= radius**2

# Calculate the protocol for each x0 and p0 value, applying the mask
protocol = np.zeros_like(x0)
for i in range(len(x0_range)):
    for j in range(len(p0_range)):
        if circle_mask[i, j]:  # Only calculate protocol inside the circle
            protocol[i, j] = S(x0[i, j], p0[i, j])

# Mask out the values outside the circular region by setting them to np.nan
protocol = np.where(circle_mask, protocol, np.nan)

# Plotting the heatmap within a circular frame
plt.figure(figsize=(10, 8))
plt.pcolormesh(x0, p0, protocol, shading='auto', cmap = 'cividis')
plt.colorbar(label='Score')
plt.xlabel('x₀')
plt.ylabel('p₀')
#plt.title('Maximum and Minimum Score of the Classical Case of the Protocol')
plt.xlim(-radius, radius)
plt.ylim(-radius, radius)
plt.gca().set_aspect('equal', adjustable='box')  # Set the aspect of the plot to be equal
plt.show()