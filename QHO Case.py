#slightly updated code from week11, works well.

import numpy as np
import math

def generate_Lminus_matrix(j):
    # Initialize the matrix
    matrix = []
    
    # Determine the range for m based on j
    # We calculate the maximum m which could be either j or the closest lower integer to j
    max_m = int(2 * j)  # This adjusts m's range for half-integer values of j
    m_range = []
    hbar = 1  # in Joule seconds

    for m_num in range(-int(j*2), int(j*2)+1, 2):
        if m_num % 2 != 0:  # Exclude 0 and include only fractions
            m = m_num / 2.0  # Convert m_num to a fraction
            m_range.append(m)
        else:
            m = m_num / 2 
            m_range.append(m)
    
    # Possible values of m range based on adjusted j value
    for m1 in m_range:  # iterating through Black M values, adjusted for half-integers
        row = []
        for m2 in m_range:  # iterating through Red M values, adjusted for half-integers
            # Apply the rule, adjusting for the fact that j is now a float
            if m1 == m2 +1:  # Adjusted due to the range change for half-integer j
                row.append(math.sqrt((j-m2)*(j+m2+1))* hbar)  # Keeping the example value as sqrt(2)
            else:
                row.append(0)
        matrix.append(row)

    return np.matrix(matrix)


# Define constants
T = np.pi
K = 200

# Define the time points for different scenarios
time_points = np.array([np.pi*k/K for k in range(K)])

j = 100
# Generate the matrix for j = 0.5
Lminus = generate_Lminus_matrix(j)

# Calculate Lplus as the complex conjugate and transpose of Lminus
Lplus = np.conj(np.transpose(Lminus))

# Print the matrices
#print("Lplus:")
#print(Lplus)

#print("\nLminus:")
#print(Lminus)

# Calculate Lx and Ly
Lx = (Lplus + Lminus) / 2
Ly = (Lplus - Lminus) / (2j)

#print('Lx+Ly = ', Lx+Ly) 
#print('Ly = ', Ly)

# Calculate Lx(theta)
def L_x(m):
    return (np.cos(m) * Lx) + (np.sin(m) * Ly )

def bra_L_x(m):
    return np.conj(L_x(m)).T

#print(np.round(L_x(j),3))
#print('')

#print(np.round(bra_L_x(j),3))
#print('')

def func(A, delta): 
    eigenvalues, eigenvectors =  np.linalg.eig(A)
    U = eigenvectors
    D = np.diag(eigenvalues)
    def f(x):
        return np.abs(x) <= delta/2 
    
    # Applying f(x) to the diagonal elements of D (eigenvalues matrix)
    f_D = np.diag([f(diag_element) for diag_element in np.diag(D)])
    
    # Computing f(A)
    f_A = U.dot(f_D).dot(np.conj(U).T)

    return f_A

#print(func(np.matrix([[1,0],[0,1]]),3))
# Initialize an empty list to accumulate Prob values
Prob_list = []

for i in range(len(time_points)):
    Prob_list.append((func(L_x(time_points[i]), 1/4)))

#print(Prob_list)

x = np.zeros(np.shape(Lx))
for i in range(len(Prob_list)): 
    if i%2 != 0:
        x = x - Prob_list[i]
    else:
        x = x + Prob_list[i]

x = x/K

print(np.round(x,3))
#delta = np.linspace(0, 2*j, 100)


#finding maximum eigenvalue: 
eigenvalues, eigenvectors =  np.linalg.eig(x)
eig_vect = eigenvectors
#eig_val = np.diag(eigenvalues)


eig_val = eigenvalues
print(np.round(eigenvalues, 3))
