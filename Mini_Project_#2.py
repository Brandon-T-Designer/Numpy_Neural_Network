import numpy as np
import pandas
import matplotlib.pyplot as plt
import pickle
import sys
import numpy
from numpy.matlib import zeros
numpy.set_printoptions(threshold=sys.maxsize)
import os

# Compute the tanh function, given a 3x1 input x
def theta(x):
    # computes the theta values for an input vector x
    return (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))

# Compute the 3x1 Derivative of the tanh function, given a 3x1 input x
def derivative_theta(x):
    return (1 - (theta(x)**2))

# Compute the scalar f_w for a given 3x1 vector x and a 16x1 vector w
def f_w(x, w):
    # Compute Theta Values
    Theta_1 = theta(w[2 - 1] * x[1 - 1] + w[3 - 1] * x[2 - 1] + w[4 - 1] * x[3 - 1] + w[5 - 1])
    Theta_2 = theta(w[7 - 1] * x[1 - 1] + w[8 - 1] * x[2 - 1] + w[9 - 1] * x[3 - 1] + w[10 - 1])
    Theta_3 = theta(w[12 - 1] * x[1 - 1] + w[13 - 1] * x[2 - 1] + w[14 - 1] * x[3 - 1] + w[15 - 1])

    # Compute f_w
    f_w = w[1 - 1] * Theta_1 + w[6 - 1] * Theta_2 + w[11 - 1] * Theta_3 + w[16 - 1]

    return f_w

# Computes the 16x1 Gradient vector of theta, given a 16x1 w vector and a 3x1 x input
def Gradient(w, x):
    # Ensure w and x are NumPy arrays
    w = np.asarray(w).reshape(-1, 1)
    x = np.asarray(x).reshape(-1, 1)

    # Create the Gradient vector
    Grad_size = w.shape[0]
    Gradient_theta = np.zeros((Grad_size, 1))

    # Construct Vectors a_i and b_i
    a = np.array([w[0][0], w[5][0], w[10][0]]).reshape(1, 3)
    b = np.array([w[4][0], w[9][0], w[14][0]]).reshape(1, 3)

    for m in range(0, Grad_size):

        # Find the current alpha value
        i = 0
        if (1 <= m <= 4):
            i = 0
        elif (6 <= m <= 9):
            i = 1
        elif (11 <= m <= 14):
            i = 2

        # Find x[j]
        j = 0
        if ((m == 1) or (m == 6) or (m == 11)):
            j = 0
        elif ((m == 2) or (m == 7) or (m == 12)):
            j = 1
        elif ((m == 3) or (m == 8) or (m == 13)):
            j = 2

        # Compute the Gradient

        # w_m = a_i
        if ((m == 0) or (m == 5) or (m == 10)):
            Gradient_theta[m, 0] = theta((a @ x).item() + (b[i, 0]))

        # w_m = B
        elif (m == 15):
            Gradient_theta[m, 0] = 1

        else:

            # W[m] = b_i
            if ((m == 4)):
                Gradient_theta[m, 0] = (a[0, i]) * derivative_theta((a @ x).item() + (b[0, i]))

            elif ((m == 9)):
                Gradient_theta[m, 0] = (a[0, i]) * derivative_theta((a @ x).item() + (b[0, i]))

            elif ((m == 14)):
                Gradient_theta[m, 0] = (a[0, i]) * derivative_theta((a @ x).item() + (b[0, i]))

            # W[m] is none of the above, and thus one of a_i_j
            else:
                Gradient_theta[m, 0] = (a[0, i]) * derivative_theta((a @ x).item() + (b[0, i])) * (x[j, 0])

    return Gradient_theta

# Computes the Nx16 Dr(w) matrix for a given 3xN X input,and a 3x1 w,
def Jacobian(x_full, w):
    Dr_w = np.zeros((x_full.shape[1], w.shape[0]))

    #print("w:\n", w.shape[0],w.shape[1])
    #print("x_full\n",x_full.shape[0],x_full.shape[1])
    #print("Dr_w:\n",Dr_w)

    for j in range(0, x_full.shape[1]):
        Grad = Gradient(w, x_full[:, j]).transpose()
        Dr_w[j, :] = Grad
    return Dr_w

# Computes Nx1 r_w vector, given a 16x1 w vector a 3xN X input, and a Nx1 y input
def r_w(x_full, y, w):
    N = y.shape[0]
    r_w = np.zeros((N, 1))

    #Old Diagnostic Prints
    """ 
    print("x_full\n", x_full)
    print(end = "\n")
 
    print("y\n", y)
    print(end = "\n")
 
    print("r_w\n", r_w)
    print(end = "\n")
    """

    for i in range(0, N):
        r_w[i, 0] = (f_w(x_full[:, i], w) - y[i]).item()

    return r_w

# Computes the loss function scalar l_w given a 3xN X input matrix, a Nx1 y value, a 16x1 w input, and a scalar lambda lam
def l_w(x_full, y, w, lam):
    R = r_w(x_full, y, w)
    R_squared = R.T@R

    l_w = R_squared + lam*(w.T@w)

    return l_w.item()

# Compute the RMSE Error using a Nx1 testing vector y _actual, a 3xN X input matrix, and a 16x1 weights vector
def RMSE(y_actual, x_full, w):

    #Create y_predicted Vector
    N = x_full.shape[1]
    y_predicted = np.zeros((N, 1))

    #fill out values in y_predicted
    for i in range(0, N):
        y_predicted[i, 0] = (f_w(x_full[:, i], w)).item()

    #Compute RMSE
    y_error = y_actual - y_predicted
    RMSE = np.sqrt((y_error.T@y_error)/(y_error.shape[0]))

    return RMSE.item()

#(w.shape[0] = 16) Compute the Current (N+16)x1 h(w) vector given a 3xN X input matrix, a Nx1 y input vector, a 16x1 w vector, and a lambda lam
def Find_h_w(x_full, y, w, lam):

    # Find Dh_w
    Dr_w = Jacobian(x_full, w)
    lam_identity = (np.sqrt(lam)) * np.identity(Dr_w.shape[1])
    Dh_w = np.vstack((Dr_w, lam_identity))
    # print("Dh_w\n",Dh_w, Dh_w.shape[0])

    # Find b Vector
    b_top = r_w(x_full, y, w)
    b_bottom = np.zeros((w.shape[0], 1))
    b = np.vstack((b_top, b_bottom))
    h_w = (Dh_w @ w) + b

    # print("h_w\n", h_w)
    return h_w

# Computes w_(t+1) given a 3xN X input matrix, a Nx1 y input vector, a 16x1 weights vector, a scalar lambda lam, and a scalar step size gamma
def Find_next_weight_vector(x_full, y, current_weights, lam, gamma):
    # Find Dh_w
    Dr_w = Jacobian(x_full, current_weights)
    lam_identity = (np.sqrt(lam)) * np.identity(Dr_w.shape[1])
    Dh_w = np.vstack((Dr_w, lam_identity))

    # Create gamma I
    gamma_identity = (np.sqrt(gamma)) * np.identity(Dh_w.shape[1])

    # Create A
    A = np.vstack((Dh_w, gamma_identity))
    # print("A\n", A.shape[0], A.shape[1] )

    # Find b Vector
    b_top = r_w(x_full, y, current_weights)
    b_bottom = np.zeros((16, 1))
    b = np.vstack((b_top, b_bottom))

    # Create Stacked Weights vector
    z = np.vstack((-b, (np.sqrt(gamma)) * current_weights))
    # print("z\n", z.shape[0],z.shape[1])

    # Compute the next set of weights
    w_next = ((np.linalg.pinv(A.T @ A)) @ A.T) @ z

    #Diagnostic Print
    """"
    print("z\n", z)
    print("A\n", A)
    print("w_next\n", w_next)
    """

    return w_next

# Computes (1xi Iteration_Indices_Vector, 1xi Loss_Values_Vector, 16x1 trained weights vector) given a 3xN x input vector, a Nx1 y input vector, a 16x1 weights vector, a scalar lambda lam, and a scalar Gamma
def Compute_Levenberg_Marquardt_Data(x_full, y, start_weights, lam, gamma, stop_threshold, stop_i):

    # Initalize Data Vectors for Graphing
    Iteration_Indices_Vector = []
    Loss_Values_Vector = []

    #Initalize Weights and Gamma
    current_weights = start_weights
    current_gamma = gamma

    #Set up Initial Loop Conditions
    i = 0
    trigger = True
    Smallest_Mag_Squared_h_w = np.dot((Find_h_w(x_full, y, current_weights, lam)).T,(Find_h_w(x_full, y, current_weights, lam)))
    while ((i < stop_i) and trigger):

        #Compute Loss Function based on the current set of weights
        current_loss = l_w(x_full, y, current_weights, lam)

        #Append my data vectors(for graphing) with the necessary Data
        Iteration_Indices_Vector = np.append(Iteration_Indices_Vector, (i+1))
        Loss_Values_Vector = np.append(Loss_Values_Vector, current_loss)



        # Find the Outputs when given the current set of weights
        h_w_t = Find_h_w(x_full, y, current_weights, lam)

        # Find the new output when given the updated set of weights
        next_weights = Find_next_weight_vector(x_full, y, current_weights, lam, current_gamma)
        h_w_t_plus_1 = Find_h_w(x_full, y, next_weights, current_gamma)

        # Update Weights by comparing old vs new Magnitudes
        Mag_Squared_h_w_t = np.dot(h_w_t.T, h_w_t).item()
        Mag_Squared_h_w_t_plus_1 = np.dot(h_w_t_plus_1.T, h_w_t_plus_1).item()

        if (Mag_Squared_h_w_t_plus_1 < Mag_Squared_h_w_t):

            # Update Weights
            current_weights = next_weights

            # Update Lambda
            current_gamma = 0.8 * current_gamma

        else:
            # Update Lambda
            current_gamma = 2 * current_gamma

        #Print the Data about the Current Iteration
        #print(f"Loss is: {current_loss}, Mag_Squared_h_w_t_plus_1: {Mag_Squared_h_w_t_plus_1}, iteration: {i}", end="\n")

        # Stopping Criteria
        #This module will stop the algorithm if the Magnitude of the residual grows beyond a reasonable limit (Which tends to happen for some reason)

        #Update the Smallest Recorded h_w_squared value
        if ( Mag_Squared_h_w_t_plus_1 < Smallest_Mag_Squared_h_w):
            Smallest_Mag_Squared_h_w = Mag_Squared_h_w_t_plus_1

        # Check if the updated h_w_squared is bigger than the smallest recorded h_w_squared value times stop_threshold
        elif (Mag_Squared_h_w_t_plus_1 > (stop_threshold*Smallest_Mag_Squared_h_w)):
            trigger = False

            #Diagnostic Print
            """
            print(f"     Stopped at iteration #{i+1}", end="\n")
            print(f"     Stop Condition: h_w_t_plus_1 has runaway growth: ", end ="\n")
            print("     Final Model Training Loss Error is:", current_loss, end="\n")
            print("     Smallest residual found is:",Smallest_Mag_Squared_h_w, end ="\n")
            print("     Final residual found is:", Mag_Squared_h_w_t_plus_1, end ="\n")
            print(end = "\n")
            """

        #Obslete Stopping Criteria (reliant on manual setting of a stop_threshold where (h_w_t_plus_1 < stop_threshold))
        """
        # Small h(w) stop Magnitude
        if (Mag_Squared_h_w_t_plus_1 <= stop_threshold):
            trigger = False    
            
            #Diagnostic Prints
            print(f"     Stopped at iteration #{i+1}", end="\n")
            print(f"    Stop Condition: h_w_t_plus_1 is below the threshold: ", end ="\n")
            print("     Final Model Training Loss Error is:", current_loss, end="\n")
            #print("     Smallest h(w) found is:",Smallest_Mag_Squared_h_w, end ="\n")
            #print("     Final h(w) found is:", Mag_Squared_h_w_t_plus_1, end ="\n")
            print(end = "\n")
        """

        # Iterator to stop my search algorithm if it drags on for too long
        i = i + 1

        #Diagnostic print function

        if (i == stop_i):

            """
            print(f"     Stopped at iteration #{i+1}", end="\n")
            print(f"    Stop Condition: i = stop_i: i = {stop_i}", end="\n")
            print("     Final Model Training Loss Error is:", current_loss, end="\n")
            print("     Smallest h(w) found is:",Smallest_Mag_Squared_h_w, end ="\n")
            print("     Final h(w) found is:", Mag_Squared_h_w_t_plus_1, end ="\n")
            print(end = "\n")
            """

    return (Iteration_Indices_Vector, Loss_Values_Vector, current_weights)

#Creates Plotted Data for a given X_Input and Y_Input, returns (fig, axs) if user wishes to re-access the graph in python
def plot_levenberg_marquardt(X_Input, y_Input, start_weights, lam, gamma, stop_threshold, stop_i, figure_title):

    y = y_Input
    # Different Values of Lambda
    (f_x_i_Vals_lam1, f_x_Loss_Vals_lam1, Computed_Weights_lam1) = Compute_Levenberg_Marquardt_Data(X_Input, y, start_weights, lam, gamma, stop_threshold, stop_i)
    (f_x_i_Vals_lam2, f_x_Loss_Vals_lam2, Computed_Weights_lam2) = Compute_Levenberg_Marquardt_Data(X_Input, y, start_weights, 0.1, gamma, stop_threshold, stop_i)
    (f_x_i_Vals_lam3, f_x_Loss_Vals_lam3, Computed_Weights_lam3) = Compute_Levenberg_Marquardt_Data(X_Input, y, start_weights, 1, gamma, stop_threshold, stop_i)
    (f_x_i_Vals_lam4, f_x_Loss_Vals_lam4, Computed_Weights_lam4) = Compute_Levenberg_Marquardt_Data(X_Input, y, start_weights, 10, gamma, stop_threshold, stop_i)

    # Different Values of Gamma
    (f_x_i_Vals_gamma1, f_x_Loss_Vals_gamma1, Computed_Weights_gamma1) = Compute_Levenberg_Marquardt_Data(X_Input, y, start_weights, lam, 0.01, stop_threshold, stop_i)
    (f_x_i_Vals_gamma2, f_x_Loss_Vals_gamma2, Computed_Weights_gamma2) = Compute_Levenberg_Marquardt_Data(X_Input, y, start_weights, lam, 0.1, stop_threshold, stop_i)
    (f_x_i_Vals_gamma3, f_x_Loss_Vals_gamma3, Computed_Weights_gamma3) = Compute_Levenberg_Marquardt_Data(X_Input, y, start_weights, lam, gamma, stop_threshold, stop_i)
    (f_x_i_Vals_gamma4, f_x_Loss_Vals_gamma4, Computed_Weights_gamma4) = Compute_Levenberg_Marquardt_Data(X_Input, y, start_weights, lam, 10, stop_threshold, stop_i)

    # Different Starting Weights
    (f_x_i_Vals_weights1, f_x_Loss_Vals_weights1, Computed_Weights_weights1) = Compute_Levenberg_Marquardt_Data(X_Input, y, 2*start_weights, lam, gamma, stop_threshold, stop_i)
    (f_x_i_Vals_weights2, f_x_Loss_Vals_weights2, Computed_Weights_weights2) = Compute_Levenberg_Marquardt_Data(X_Input, y, 4*start_weights, lam, gamma, stop_threshold, stop_i)
    (f_x_i_Vals_weights3, f_x_Loss_Vals_weights3, Computed_Weights_weights3) = Compute_Levenberg_Marquardt_Data(X_Input, y, 6*start_weights, lam, gamma, stop_threshold, stop_i)
    (f_x_i_Vals_weights4, f_x_Loss_Vals_weights4, Computed_Weights_weights4) = Compute_Levenberg_Marquardt_Data(X_Input, y, 8*start_weights, lam, gamma, stop_threshold, stop_i)

    #Truncate elements for plotting if they exceed 80 elements
    f_x_i_Vals_lam1 = f_x_i_Vals_lam1[:80]
    f_x_i_Vals_lam2 = f_x_i_Vals_lam2[:80]
    f_x_i_Vals_lam3 = f_x_i_Vals_lam3[:80]
    f_x_i_Vals_lam4 = f_x_i_Vals_lam4[:80]
    f_x_Loss_Vals_lam1 = f_x_Loss_Vals_lam1[:80]
    f_x_Loss_Vals_lam2 = f_x_Loss_Vals_lam2[:80]
    f_x_Loss_Vals_lam3 = f_x_Loss_Vals_lam3[:80]
    f_x_Loss_Vals_lam4 = f_x_Loss_Vals_lam4[:80]
    f_x_i_Vals_gamma1 = f_x_i_Vals_gamma1[:80]
    f_x_i_Vals_gamma2 = f_x_i_Vals_gamma2[:80]
    f_x_i_Vals_gamma3 = f_x_i_Vals_gamma3[:80]
    f_x_i_Vals_gamma4 = f_x_i_Vals_gamma4[:80]
    f_x_i_Vals_weights1 = f_x_i_Vals_weights1[:80]
    f_x_i_Vals_weights2 = f_x_i_Vals_weights2[:80]
    f_x_i_Vals_weights3 = f_x_i_Vals_weights3[:80]
    f_x_i_Vals_weights4 = f_x_i_Vals_weights4[:80]

    # Create a unique figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 20))

    # Subplot 1: Different Values of Lambda
    axs[0].plot(f_x_i_Vals_lam1, f_x_Loss_Vals_lam1, color='red', marker='o', label='λ = 10^(-5)')
    axs[0].plot(f_x_i_Vals_lam2, f_x_Loss_Vals_lam2, color='orange', marker='o', label='λ = 0.1')
    axs[0].plot(f_x_i_Vals_lam3, f_x_Loss_Vals_lam3, color='green', marker='o', label='λ = 1')
    axs[0].plot(f_x_i_Vals_lam4, f_x_Loss_Vals_lam4, color='blue', marker='o', label='λ = 10')
    axs[0].set_xlabel("Iteration Number")
    axs[0].set_ylabel("Loss Magnitude")
    axs[0].set_title("Loss vs Iterations for Different λ Values")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Different Values of Gamma
    axs[1].plot(f_x_i_Vals_gamma1, f_x_Loss_Vals_gamma1, color='red', marker='o', label='γ = 0.01')
    axs[1].plot(f_x_i_Vals_gamma2, f_x_Loss_Vals_gamma2, color='orange', marker='o', label='γ = 0.1')
    axs[1].plot(f_x_i_Vals_gamma3, f_x_Loss_Vals_gamma3, color='green', marker='o', label='γ = 1')
    axs[1].plot(f_x_i_Vals_gamma4, f_x_Loss_Vals_gamma4, color='blue', marker='o', label='γ = 10')
    axs[1].set_xlabel("Iteration Number")
    axs[1].set_ylabel("Loss Magnitude")
    axs[1].set_title("Loss vs Iterations for Different γ Values")
    axs[1].legend()
    axs[1].grid(True)

    # Subplot 3: Different Starting Weights
    axs[2].plot(f_x_i_Vals_weights1, f_x_Loss_Vals_weights1, color='red', marker='o', label='Weights = 2*ones((16,1))')
    axs[2].plot(f_x_i_Vals_weights2, f_x_Loss_Vals_weights2, color='orange', marker='o', label='Weights = 4*ones((16,1))')
    axs[2].plot(f_x_i_Vals_weights3, f_x_Loss_Vals_weights3, color='green', marker='o', label='Weights = 6*ones((16,1))')
    axs[2].plot(f_x_i_Vals_weights4, f_x_Loss_Vals_weights4, color='blue', marker='o', label='Weights = 8*ones((16,1))')
    axs[2].set_xlabel("Iteration Number")
    axs[2].set_ylabel("Loss Magnitude")
    axs[2].set_title("Loss vs Iterations for Different Starting Weights")
    axs[2].legend()
    axs[2].grid(True)

    # Automatically adjust layout
    plt.tight_layout(pad=3.5)  # Add padding between subplots and figure edges
    # Manually fine-tune spacing
    plt.subplots_adjust(hspace=0.6, top = 0.88, bottom = 0.15)  # Adjust spacing between subplots, top, and bottom
    # Add Title
    fig.suptitle(figure_title, fontsize = 25, fontweight='bold', y=0.95)  # Adjust title position

    # Print last values of f_x_Loss_Vals
    print(f"Last Loss Value for Different λ:")
    print(f"  λ = 10^(-5): {f_x_Loss_Vals_lam1[-1]}")
    print(f"  λ = 0.1: {f_x_Loss_Vals_lam2[-1]}")
    print(f"  λ = 1: {f_x_Loss_Vals_lam3[-1]}")
    print(f"  λ = 10: {f_x_Loss_Vals_lam4[-1]}\n")

    print(f"Last Loss Value for Different γ:")
    print(f"  γ = 0.01: {f_x_Loss_Vals_gamma1[-1]}")
    print(f"  γ = 0.1: {f_x_Loss_Vals_gamma2[-1]}")
    print(f"  γ = 1: {f_x_Loss_Vals_gamma3[-1]}")
    print(f"  γ = 10: {f_x_Loss_Vals_gamma4[-1]}\n")

    print(f"Last Loss Value for Different Starting Weights:")
    print(f"  Weights = 2*ones: {f_x_Loss_Vals_weights1[-1]}")
    print(f"  Weights = 4*ones: {f_x_Loss_Vals_weights2[-1]}")
    print(f"  Weights = 6*ones: {f_x_Loss_Vals_weights3[-1]}")
    print(f"  Weights = 8*ones: {f_x_Loss_Vals_weights4[-1]}\n")

    return fig, axs

#Makes a custom Random rowsxN Input Matrix, by taking in scalars rows, N, and T(upper and lower bound for random variables)
def Custom_Random_Matrix_Maker(rows, N, T):
    X = np.random.uniform(low=-T, high=T, size=(rows, N))

    return X

#returns a 1xN test y vector where each y_i = f(x_i), with f(x) being my custom nonlinear function which dots an x input by itself
def f(x_full):
    N = x_full.shape[1]

    y = zeros((N,1))
    for i in range (0, N):
        y[i,0] = x_full[0,i]**2 + x_full[1,i]**2 + x_full[2,i]**2

    return y

#returns a 1xN y vector where each y_i = g(x_i) with g(x) being the nonlinear function givin in our Mini project
def g(x_full):
    N = x_full.shape[1]

    y = zeros((N,1))
    for i in range (0, N):
        y[i,0] = x_full[0,i]*x_full[1,i] + x_full[2,i]

    return y

#returns a 1xN y vector where each y_i = g_noisy(x_i), with g_noisy(x_i) being our g(x) function with a specified added noise E
def g_noisy(x_full,E):
    N = x_full.shape[1]

    y = zeros((N,1))
    for i in range (0, N):
        y[i,0] = x_full[0,i]*x_full[1,i] + x_full[2,i] + np.random.uniform(-E, E)

    return y

#Old Way to generate Data
"""
#3x500 Matrix Maker
X_3x500 = Custom_Random_Matrix_Maker(3, 500, 1)

#3x100 Matrix Maker
X_3x100_01 = Custom_Random_Matrix_Maker(3, 100, 0.1)
X_3x100_05 = Custom_Random_Matrix_Maker(3, 100, 0.5)
X_3x100_1 = Custom_Random_Matrix_Maker(3, 100, 1)
X_3x100_10 = Custom_Random_Matrix_Maker(3, 100, 10)

#g(x) y values
y_g_x = g(X_3x500)

#f(x) y values
y_f_x = f(X_3x500)

#Noise values for g(x)
y_noisy01 = g_noisy(X_3x500,0.1)
y_noisy1 = g_noisy(X_3x500,1)
y_noisy5 = g_noisy(X_3x500,5)
y_noisy10 = g_noisy(X_3x500,10)
"""

#New Pickle Way
# Get the current working directory
current_project_path = os.getcwd()
# Print the path
print(f"The current Python project path is: {current_project_path}")

# Pickle Storage for data
pickle_file_path = "MyData.pkl"
# Save Generated Data as a plickle File
#Comment out this code block in order to store the values of the current iteration's testing data (But make sure to move your current test data to a different directory)

with open("MyData.pkl", "wb") as file:

    #3x500 Matrix Maker
    X_3x500 = Custom_Random_Matrix_Maker(3, 500, 1)

    #3x100 Matrix Maker
    X_3x100_01 = Custom_Random_Matrix_Maker(3, 100, 0.1)
    X_3x100_05 = Custom_Random_Matrix_Maker(3, 100, 0.5)
    X_3x100_1 = Custom_Random_Matrix_Maker(3, 100, 1)
    X_3x100_10 = Custom_Random_Matrix_Maker(3, 100, 10)

    #g(x) y values
    y_g_x = g(X_3x500)

    #f(x) y values
    y_f_x = f(X_3x500)

    #Noise values for g(x)
    y_noisy01 = g_noisy(X_3x500,0.1)
    y_noisy1 = g_noisy(X_3x500,1)
    y_noisy5 = g_noisy(X_3x500,5)
    y_noisy10 = g_noisy(X_3x500,10)

    pickle.dump({
        "X_3x500": X_3x500,
        "X_3x100_01": X_3x100_01,
        "X_3x100_05": X_3x100_05,
        "X_3x100_1": X_3x100_1,
        "X_3x100_10": X_3x100_10,
        "y_g_x": y_g_x,
        "y_f_x": y_f_x,
        "y_noisy01": y_noisy01,
        "y_noisy1": y_noisy1,
        "y_noisy5": y_noisy5,
        "y_noisy10": y_noisy10
    }, file)


# Load all vectors from the pickle file
with open(pickle_file_path, "rb") as file:
    MyData = pickle.load(file)

# Convert dictionary keys into global variables
for key, value in MyData.items():
    globals()[key] = value


#Default Initialization
start_weights = np.ones((16,1))
lam = 10**-5
gamma = 1
stop_threshold = 100
stop_i = 1000

# Task 3a
print("Task 3a\n")
(fig1, axs1) = plot_levenberg_marquardt(X_3x500, y_g_x, start_weights, lam, gamma, stop_threshold, stop_i, "Training Loss Values For g(x)")

#Task 3b
print("\n")
print("Task 3b\n")

print("Different Lambdas, and T Values\n")
(I_b1, X_b1, w_b1) = Compute_Levenberg_Marquardt_Data(X_3x500, y_g_x, start_weights, 0.5, gamma, stop_threshold, stop_i)
(I_b1_2, X_b1_2, w_b1_2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_g_x, start_weights, 1, gamma, stop_threshold, stop_i)
(I_b2, X_b2, w_b2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_g_x, start_weights, 2, gamma, stop_threshold, stop_i)
(I_b2_2, X_b2_2, w_b2_2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_g_x, start_weights, 5, gamma, stop_threshold, stop_i)

print(f"Training RMSE values for different λ")
print(f"  λ = 0.5: {RMSE(g(X_3x500), X_3x500, w_b1)}")
print(f"  λ = 1: {RMSE(g(X_3x500), X_3x500, w_b1_2)}")
print(f"  λ = 2: {RMSE(g(X_3x500), X_3x500, w_b2)}")
print(f"  λ = 5: {RMSE(g(X_3x500), X_3x500, w_b2_2)}\n")

# Print Testing RMSE results for λ = 0.5
print(f"Testing RMSE values for λ = 0.5 and T = [0.1, 0.5, 1, 10]:")
print(f"  T = 0.1: {RMSE(g(X_3x100_01), X_3x100_01, w_b1)}")
print(f"  T = 0.5: {RMSE(g(X_3x100_05), X_3x100_05, w_b1)}")
print(f"  T = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_b1)}")
print(f"  T = 10: {RMSE(g(X_3x100_10), X_3x100_10, w_b1)}\n")

# Print Testing RMSE results for λ = 1
print(f"Testing RMSE values for λ = 1 and T = [0.1, 0.5, 1, 10]:")
print(f"  T = 0.1: {RMSE(g(X_3x100_01), X_3x100_01, w_b1_2)}")
print(f"  T = 0.5: {RMSE(g(X_3x100_05), X_3x100_05, w_b1_2)}")
print(f"  T = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_b1_2)}")
print(f"  T = 10: {RMSE(g(X_3x100_10), X_3x100_10, w_b1_2)}\n")

# Print Testing RMSE results for λ = 2
print(f"Testing RMSE values for λ = 2 and T = [0.1, 0.5, 1, 10]:")
print(f"  T = 0.1: {RMSE(g(X_3x100_01), X_3x100_01, w_b2)}")
print(f"  T = 0.5: {RMSE(g(X_3x100_05), X_3x100_05, w_b2)}")
print(f"  T = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_b2)}")
print(f"  T = 10: {RMSE(g(X_3x100_10), X_3x100_10, w_b2)}\n")

# Print Testing RMSE results for λ = 5
print(f"\nTesting RMSE values for λ = 5 and T = [0.1, 0.5, 1, 10]:")
print(f"  T = 0.1: {RMSE(g(X_3x100_01), X_3x100_01, w_b2_2)}")
print(f"  T = 0.5: {RMSE(g(X_3x100_05), X_3x100_05, w_b2_2)}")
print(f"  T = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_b2_2)}")
print(f"  T = 10: {RMSE(g(X_3x100_10), X_3x100_10, w_b2_2)}")
print("\n")

#Task 3c

print("Task 3c\n")
print("Task 3c_a")
(fig2, axs2) = plot_levenberg_marquardt(X_3x500, y_f_x, start_weights, lam, gamma, stop_threshold, stop_i, "Training Loss Values For f(x)")

print("Task 3c_b")
print("Different Lambdas, and T Values for f(x)")
(I_c1, X_c1, w_c1) = Compute_Levenberg_Marquardt_Data(X_3x500, y_f_x, start_weights, 0.5, gamma, stop_threshold, stop_i)
(I_c1_2, X_c1_2, w_c1_2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_f_x, start_weights, 1, gamma, stop_threshold, stop_i)
(I_c2, X_c2, w_c2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_f_x, start_weights, 2, gamma, stop_threshold, stop_i)
(I_c2_2, X_c2_2, w_c2_2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_f_x, start_weights, 5, gamma, stop_threshold, stop_i)

print(f"\nTraining RMSE values for different λ")
print(f"  λ = 0.5: {RMSE(f(X_3x500), X_3x500, w_c1)}")
print(f"  λ = 1: {RMSE(f(X_3x500), X_3x500, w_c1_2)}")
print(f"  λ = 2: {RMSE(f(X_3x500), X_3x500, w_c2)}")
print(f"  λ = 5: {RMSE(f(X_3x500), X_3x500, w_c2_2)}\n")

# Print RMSE results for λ = 0.5
print(f"\nTesting RMSE values for λ = 0.5 and T = [0.1, 0.5, 1, 10]:")
print(f"  T = 0.1: {RMSE(f(X_3x100_01), X_3x100_01, w_c1)}")
print(f"  T = 0.5: {RMSE(f(X_3x100_05), X_3x100_05, w_c1)}")
print(f"  T = 1: {RMSE(f(X_3x100_1), X_3x100_1, w_c1)}")
print(f"  T = 10: {RMSE(f(X_3x100_10), X_3x100_10, w_c1)}\n")

# Print RMSE results for λ = 1
print(f"\nTesting RMSE values for λ = 1 and T = [0.1, 0.5, 1, 10]:")
print(f"  T = 0.1: {RMSE(f(X_3x100_01), X_3x100_01, w_c1_2)}")
print(f"  T = 0.5: {RMSE(f(X_3x100_05), X_3x100_05, w_c1_2)}")
print(f"  T = 1: {RMSE(f(X_3x100_1), X_3x100_1, w_c1_2)}")
print(f"  T = 10: {RMSE(f(X_3x100_10), X_3x100_10, w_c1_2)}\n")

# Print RMSE results for λ = 2
print(f"\nTesting RMSE values for λ = 2 and T = [0.1, 0.5, 1, 10]:")
print(f"  T = 0.1: {RMSE(f(X_3x100_01), X_3x100_01, w_c2)}")
print(f"  T = 0.5: {RMSE(f(X_3x100_05), X_3x100_05, w_c2)}")
print(f"  T = 1: {RMSE(f(X_3x100_1), X_3x100_1, w_c2)}")
print(f"  T = 10: {RMSE(f(X_3x100_10), X_3x100_10, w_c2)}\n")

# Print RMSE results for λ = 5
print(f"\nTesting RMSE values for λ = 5 and T = [0.1, 0.5, 1, 10]:")
print(f"  T = 0.1: {RMSE(f(X_3x100_01), X_3x100_01, w_c2_2)}")
print(f"  T = 0.5: {RMSE(f(X_3x100_05), X_3x100_05, w_c2_2)}")
print(f"  T = 1: {RMSE(f(X_3x100_1), X_3x100_1, w_c2_2)}")
print(f"  T = 10: {RMSE(f(X_3x100_10), X_3x100_10, w_c2_2)}")
print("\n")
#Task 3d
print("Task 3d\n")

print("Task 3d_a")
(fig3, axs3) = plot_levenberg_marquardt(X_3x500, y_noisy1, start_weights, lam, gamma, stop_threshold, stop_i, "Training Loss Values For g(x) with noise E = 1")

print("Task 3d_b\n")
print("Different Lambdas, and T Values for g(x)")
(I_d1, X_d1, w_d1) = Compute_Levenberg_Marquardt_Data(X_3x500, y_noisy1, start_weights, 0.5, gamma, stop_threshold, stop_i)
(I_d1_2, X_d1_2, w_d1_2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_noisy1, start_weights, 1, gamma, stop_threshold, stop_i)
(I_d2, X_d2, w_d2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_noisy1, start_weights, 2, gamma, stop_threshold, stop_i)
(I_d2_2, X_d2_2, w_d2_2) = Compute_Levenberg_Marquardt_Data(X_3x500, y_noisy1, start_weights, 5, gamma, stop_threshold, stop_i)

print(f"Training RMSE values for different λ")
print(f"  λ = 0.5: {RMSE(g(X_3x500), X_3x500, w_d1)}")
print(f"  λ = 1: {RMSE(g(X_3x500), X_3x500, w_d1_2)}")
print(f"  λ = 2: {RMSE(g(X_3x500), X_3x500, w_d2)}")
print(f"  λ = 5: {RMSE(g(X_3x500), X_3x500, w_d2_2)}\n")

# Print RMSE results for λ = 0.5
print(f"\nTesting RMSE values for λ = 0.5 and T = [0.1, 0.5, 1, 10] for g(x):")
print(f"  T = 0.1: {RMSE(g(X_3x100_01), X_3x100_01, w_d1)}")
print(f"  T = 0.5: {RMSE(g(X_3x100_05), X_3x100_05, w_d1)}")
print(f"  T = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_d1)}")
print(f"  T = 10: {RMSE(g(X_3x100_10), X_3x100_10, w_d1)}\n")

# Print RMSE results for λ = 1
print(f"\nTesting RMSE values for λ = 1 and T = [0.1, 0.5, 1, 10] for g(x):")
print(f"  T = 0.1: {RMSE(g(X_3x100_01), X_3x100_01, w_d1_2)}")
print(f"  T = 0.5: {RMSE(g(X_3x100_05), X_3x100_05, w_d1_2)}")
print(f"  T = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_d1_2)}")
print(f"  T = 10: {RMSE(g(X_3x100_10), X_3x100_10, w_d1_2)}\n")

# Print RMSE results for λ = 2
print(f"\nTesting RMSE values for λ = 2 and T = [0.1, 0.5, 1, 10] for g(x):")
print(f"  T = 0.1: {RMSE(g(X_3x100_01), X_3x100_01, w_d2)}")
print(f"  T = 0.5: {RMSE(g(X_3x100_05), X_3x100_05, w_d2)}")
print(f"  T = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_d2)}")
print(f"  T = 10: {RMSE(g(X_3x100_10), X_3x100_10, w_d2)}\n")

# Print RMSE results for λ = 5
print(f"\nTesting RMSE values for λ = 5 and T = [0.1, 0.5, 1, 10] for g(x):")
print(f"  T = 0.1: {RMSE(g(X_3x100_01), X_3x100_01, w_d2_2)}")
print(f"  T = 0.5: {RMSE(g(X_3x100_05), X_3x100_05, w_d2_2)}")
print(f"  T = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_d2_2)}")
print(f"  T = 10: {RMSE(g(X_3x100_10), X_3x100_10, w_d2_2)}")
print("\n")


print("Task 3d_Different noise levels\n")
print("Training and Test Errors for Different Noise Levels")
(I_d3, X_d3, w_d3) = Compute_Levenberg_Marquardt_Data(X_3x500, y_noisy01, start_weights, lam, gamma, stop_threshold, stop_i)
(I_d4, X_d4, w_d4) = Compute_Levenberg_Marquardt_Data(X_3x500, y_noisy1, start_weights, lam, gamma, stop_threshold, stop_i)
(I_d5, X_d5, w_d5) = Compute_Levenberg_Marquardt_Data(X_3x500, y_noisy5, start_weights, lam, gamma, stop_threshold, stop_i)
(I_d6, X_d6, w_d6) = Compute_Levenberg_Marquardt_Data(X_3x500, y_noisy10, start_weights, lam, gamma, stop_threshold, stop_i)

print("Training and Test Errors for Different Noise Levels:")

# Training Errors (Last Loss Values)
print("Training Errors for different Noise Values:")
print(f"  Noise Level ε = 0.1: {RMSE(g(X_3x500), X_3x500, w_d3)}")
print(f"  Noise Level ε = 1: {RMSE(g(X_3x500), X_3x500, w_d4)}")
print(f"  Noise Level ε = 5: {RMSE(g(X_3x500), X_3x500, w_d5)}")
print(f"  Noise Level ε = 10: {RMSE(g(X_3x500), X_3x500, w_d6)}\n")

# Test Errors (RMSE)
print("Test Errors for different Noise Values:")
print(f"  Noise Level ε = 0.1: {RMSE(g(X_3x100_1), X_3x100_1, w_d3)}")
print(f"  Noise Level ε = 1: {RMSE(g(X_3x100_1), X_3x100_1, w_d4)}")
print(f"  Noise Level ε = 5: {RMSE(g(X_3x100_1), X_3x100_1, w_d5)}")
print(f"  Noise Level ε = 10: {RMSE(g(X_3x100_1), X_3x100_1, w_d6)}")

plt.show()






























