# Import NumPy for numerical operations
import numpy as np
# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
# Import Matplotlib for plotting
import matplotlib.pyplot as plt
# Import a utility module for additional plotting functions
from utils import mpl as plt
from utils import cmap_ 
# Import the time module to time our training process
import time
# Ignore Warning Messages
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import solve_ivp


#%% NUMERICALL -----------------------------------------------------------
# Define the function to be approximated

# Definimos las constantes
g = 9.81  # Aceleración debido a la gravedad (m/s^2)
L = 1.0   # Longitud del péndulo (m)
theta0 = np.pi / 4  # Ángulo inicial (radianes)
omega0 = 0.0        # Velocidad angular inicial (radianes/segundo)

# Definimos el sistema de ecuaciones diferenciales
def pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Condiciones iniciales
y0 = [theta0, omega0]

# Tiempo de integración
t_span = (0, 10)  # de 0 a 10 segundos
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # puntos de tiempo para evaluar

# Resolvemos la ecuación diferencial
sol = solve_ivp(pendulum, t_span, y0, t_eval=t_eval, method='RK45')

# Extraemos las soluciones
theta = sol.y[0]
omega = sol.y[1]
t = sol.t

# Graficamos los resultados
plt.figure(figsize=(12, 6))
plt.plot(t, theta, label='θ(t) (Angular Displacement)')
plt.plot(t, omega, label='ω(t) (Angular Velocity)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Nonlinear Pendulum Solution')
plt.grid(True)
plt.show()

# # Generate training data in NumPy
# x_np = np.linspace(0, 20, 100)  # x data (numpy array), shape=(100,)
# y_np = pendulum(x_np)  # y data (numpy array), shape=(100,)
# # Convert the NumPy arrays to PyTorch tensors and add an extra dimension
# input_data = torch.from_numpy(x_np).float().unsqueeze(-1)  # Convert x data to a PyTorch tensor
# y = torch.from_numpy(y_np).float().unsqueeze(-1)  # Convert y data to a PyTorch tensor


# # Define a neural network class with three fully connected layers
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.layer1 = nn.Linear(1, 10)
#         self.layer2 = nn.Linear(10, 10)
#         self.output_layer = nn.Linear(10, 1)

#     def forward(self, x):
#         x = torch.tanh(self.layer1(x))
#         x = torch.tanh(self.layer2(x))
#         x = self.output_layer(x)
#         return x
    
# # Create an instance of the neural network
# neural_net = NeuralNetwork()

# # Define an optimizer (Adam) for training the network
# optimizer = optim.Adam(neural_net.parameters(), lr=0.01)

# # Define a loss function (Mean Squared Error) for training the network
# loss_func = nn.MSELoss()


# # Initialize a list to store the loss values
# loss_values = []

# # Start the timer
# start_time = time.time()

# # Training the neural network
# for i in range(10_001):
#     prediction = neural_net(input_data)     # input x and predict based on x
#     loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    
#     # Append the current loss value to the list
#     loss_values.append(loss.item())
    
#     if i % 1000 == 0:  # print every 100 iterations
#         print(f"Iteration {i}: Loss {loss.item()}")
    
#     optimizer.zero_grad()   # clear gradients for next train
#     loss.backward()         # backpropagation, compute gradients
#     optimizer.step()

# # Stop the timer and calculate the elapsed time
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Training time: {elapsed_time} seconds")

    
# # Save a summary of the training process to a text file
# with open("summaries/1_Simple_Function_Aproximation.txt", "w") as file:
#     file.write("Summary of Neural Network Training\n")
#     file.write("=================================\n\n")
#     file.write(f"Neural Network Architecture:\n{neural_net}\n\n")
#     file.write(f"Optimizer Used:\n{type(optimizer).__name__}\n\n")
#     file.write(f"Learning Rate:\n{optimizer.param_groups[0]['lr']}\n\n")
#     file.write(f"Number of Iterations:\n{len(loss_values)}\n\n")
#     file.write(f"Initial Loss:\n{loss_values[0]}\n\n")
#     file.write(f"Final Loss:\n{loss_values[-1]}\n\n")
#     file.write(f"Training Time:\n{elapsed_time} seconds\n\n")
    
#     # Calculate the average loss
#     average_loss = sum(loss_values) / len(loss_values)
#     file.write(f"Average Loss:\n{average_loss}\n\n")
    
#     # Find the iteration with the minimum loss
#     min_loss_value = min(loss_values)
#     min_loss_iteration = loss_values.index(min_loss_value)
#     file.write(f"Minimum Loss:\n{min_loss_value} at iteration {min_loss_iteration}\n\n")    