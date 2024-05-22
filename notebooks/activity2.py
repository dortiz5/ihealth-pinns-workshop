# Import NumPy for numerical operations
import numpy as np
# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
# Import Matplotlib for plotting
import matplotlib.pyplot as plt
# Import a utility module 
import utils 
# Import the time module to time our training process
import time
# Ignore Warning Messages
import warnings
warnings.filterwarnings("ignore")

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def relative_l2_error(u_num, u_ref):
    # Calculate the L2 norm of the difference
    l2_diff = torch.norm(u_num - u_ref, p=2)
    
    # Calculate the L2 norm of the reference
    l2_ref = torch.norm(u_ref, p=2)
    
    # Calculate L2 relative error
    relative_l2 = l2_diff / l2_ref
    return relative_l2

def plot_comparison(time, u_true, u_pred, loss):
    
    # Convert tensors to numpy arrays for plotting
    u_pred_np = u_pred.detach().numpy()

    # Create a figure with two subplots
    _, axs = plt.subplots(2, 2, figsize=(12, 6))
    
    # Plot the actual y values
    axs[0, 0].imshow(u_true, extent=[-1,1,1,0])
    axs[0, 0].set_title('Analytic solution for diffusion')
    axs[0, 0].set_xlabel(r'$x$')
    axs[0, 0].set_ylabel(r'$t$') 

    # Plot the actual y values
    axs[0, 1].imshow(u_pred_np, extent=[-1,1,1,0])
    axs[0, 1].set_title('PINN solution for diffusion')
    axs[0, 0].set_xlabel(r'$x$')
    axs[0, 0].set_ylabel(r'$t$') 

    # Plot the difference between the predicted and actual y values
    difference = np.abs(u_true - u_pred_np)
    axs[1, 0].imshow(difference, extent=[-1,1,1,0])
    axs[1, 0].set_title(r'$|u(t,x) - u_{pred}(t,x)|$')
    axs[1, 0].set_xlabel(r'$x$')
    axs[1, 0].set_ylabel(r'$t$') 
    # Display the plot
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()

    # Plot the loss values recorded during training
    # Create a figure with two subplots
    _, axs = plt.subplots(1, 1, figsize=(6, 3))
    axs.plot(loss)
    axs.set_xlabel('Iteration')
    axs.set_ylabel('Loss')
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.set_title('Training Progress')
    axs.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    l2_error = relative_l2_error(u_true.reshape(-1,1),
                            u_pred_np.reshape(-1,1))
    print(f'Relative l2 error: {l2_error}')

def grad(outputs, inputs):
    """Computes the partial derivative of an output with respect 
    to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(outputs, inputs, 
                        grad_outputs=torch.ones_like(outputs), 
                        create_graph=True,
                        retain_graph=True,  
                        )[0]
    
    
# Import NumPy for numerical operations
import numpy as np

def analytic_diffusion(x,t):
    y = np.exp(-t)*np.sin(np.pi*x)
    return y

x = np.linspace(-1, 1, 100)
t = np.linspace(0, 1, 100)

X, T = np.meshgrid(x, t)
Y = analytic_diffusion(X, T)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(Y, extent=[-1,1,1,0])
ax.set_title('Analytic solution for diffusion')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$') 
ax.legend(loc='lower left', frameon=False)
plt.tight_layout()
plt.show()




from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=100)

l_bounds = [-1, 0]
u_bounds = [ 1, 1]
domain_xt = qmc.scale(sample, l_bounds, u_bounds)

x_ten = torch.tensor(domain_xt[:, 0], requires_grad = True).float().reshape(-1,1)
t_ten = torch.tensor(domain_xt[:, 1], requires_grad = True).float().reshape(-1,1)




torch.manual_seed(123)

# training parameters
hidden_layers = [2, 10, 10, 10, 1]
learning_rate = 0.001
training_iter = 20000



# Define a loss function (Mean Squared Error) for training the network
MSE_func = nn.MSELoss()
# Define a neural network class with user defined layers and neurons
class NeuralNetwork(nn.Module):
    
    def __init__(self, hlayers):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        for i in range(len(hlayers[:-2])):
            layers.append(nn.Linear(hlayers[i], hlayers[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hlayers[-2], hlayers[-1]))
        
        self.layers = nn.Sequential(*layers)
        self.init_params
        
    def init_params(self):
        """Xavier Glorot parameter initialization of the Neural Network
        """
        def init_normal(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) # Xavier
        self.apply(init_normal)

    def forward(self, x):
        return self.layers(x)
    
    
    
# Create an instance of the neural network 
u_pinn = NeuralNetwork(hidden_layers)
nparams = sum(p.numel() for p in u_pinn.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

# Define an optimizer (Adam) for training the network
optimizer = optim.Adam(u_pinn.parameters(), lr=0.001, 
                       betas= (0.9,0.999), eps = 1e-8)



# HINT: 
def PINN_diffusion_Loss(forward_pass, x_ten, t_ten, 
             lambda1 = 1, lambda2 = 1, lambda3 = 1, lambda4 = 1):

    # ANN output, first and second derivatives
    domain = torch.cat([t_ten, x_ten], dim = 1)
    u = forward_pass(domain)
    u_t = grad(u, t_ten)
    u_x = grad(u, x_ten)
    u_xx = grad(u_x, x_ten)
    
    # PDE loss definition
    f_pde = u_t - u_xx + torch.exp(-t_ten)*(torch.sin(np.pi*x_ten)
                          -(torch.pi**2)*torch.sin(np.pi*x_ten))
    PDE_loss = lambda1 * MSE_func(f_pde, torch.zeros_like(f_pde)) 
    
    # IC loss definition
    ic = torch.cat([torch.zeros_like(t_ten), x_ten], dim = 1)
    g_ic = forward_pass(ic)
    IC_loss = lambda2 * MSE_func(g_ic, -torch.sin(torch.pi*x_ten))

    # BC x = -1 definition
    bc1 = torch.cat([t_ten, -torch.ones_like(x_ten)], dim = 1)
    h_bc1 = forward_pass(bc1)
    BC1_loss = lambda3 * MSE_func(h_bc1, torch.zeros_like(h_bc1))
    
    # BC x = 1 definition
    bc2 = torch.cat([t_ten, torch.ones_like(x_ten)], dim = 1)
    h_bc2 = forward_pass(bc2)
    BC2_loss = lambda3 * MSE_func(h_bc2, torch.zeros_like(h_bc2))
    
    return PDE_loss + IC_loss + BC1_loss + BC2_loss
    
# Initialize a list to store the loss values
loss_values = []

# Start the timer
start_time = time.time()

# Training the neural network
for i in range(training_iter):
    
    optimizer.zero_grad()   # clear gradients for next train

    # input x and predict based on x
    loss = PINN_diffusion_Loss(u_pinn, x_ten, t_ten)
    
    # Append the current loss value to the list
    loss_values.append(loss.item())
    
    if i % 1000 == 0:  # print every 100 iterations
        print(f"Iteration {i}: Loss {loss.item()}")
    
    loss.backward() # compute gradients (backpropagation)
    optimizer.step() # update the ANN weigths

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")