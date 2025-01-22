import torch
import torch.nn as nn
import numpy as np

# Define the neural network for the PINN
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3) # Output : [u,v,p]
        )

    def forward(self, x):        
        return self.net(x)

# Define the Physics-Informed Loss
def physics_loss(model, x, y, t):    
    inputs = torch.cat((x,y,t), dim=1)
    outputs = model(inputs)
    u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

    # Compute derivatives with respect to inputs
    ut  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    ux  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uy  = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(uy), create_graph=True)[0]

    vt  = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    vx  = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    vy  = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    vxx = torch.autograd.grad(vx, x, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
    vyy = torch.autograd.grad(vy, y, grad_outputs=torch.ones_like(vy), create_graph=True)[0]

    px  = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    py  = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    f1  = -torch.cos(x)*torch.sin(y)*(2*torch.cos(t)-torch.sin(t))
    f2  =  torch.cos(y)*torch.sin(x)*(2*torch.cos(t)-torch.sin(t))

    # Residuals
    momentum_u = ut+u*ux+v*uy+px-(uxx+uyy)-f1
    momentum_v = vt+u*vx+v*vy+py-(vxx+vyy)-f2
    continuity = ux + vy

    return momentum_u, momentum_v, continuity    

def boundary_loss(model, x, y, t):    
    inputs = torch.cat((x, y, t), dim=1)    
    outputs = model(inputs)
    u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

    # Apply boundary conditions (u, v and p on walls)
    bc_u = -     torch.cos(x)*torch.sin(y)*torch.cos(t)
    bc_v =       torch.sin(x)*torch.cos(y)*torch.cos(t)
    bc_p = -0.25*torch.cos(t)*torch.cos(t)*(torch.cos(2*x)+torch.cos(2*y))

    return u-bc_u, v-bc_v, p-bc_p    

def initial_loss(model, x, y, t):
    inputs = torch.cat((x, y, t), dim=1)
    outputs = model(inputs)
    u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:,2:3]

    # Initial conditions (u, v and p)
    init_u = -     torch.cos(x)*torch.sin(y)
    init_v =       torch.sin(x)*torch.cos(y)
    init_p = -0.25*(torch.cos(2*x)+torch.cos(2*y))
    
    return u-init_u, v-init_v, p-init_p    

def predict(model, x, y, t):
    inputs = torch.cat((x, y, t), dim=1)
    outputs = model(inputs)
    u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:,2:3]

    return u, v, p

# Prediciton part
model = PINN()
model.load_state_dict(torch.load("single_vortex.pt", weights_only=True))

# Prediction
x = torch.linspace(-torch.pi/2, torch.pi/2, 100)
y = torch.linspace(-torch.pi/2, torch.pi/2, 100)
X, Y = torch.meshgrid(x, y, indexing="ij")
X, Y = X.reshape(-1,1), Y.reshape(-1,1)
t    = torch.pi*torch.rand(1)*torch.ones_like(X)

U,V,p=predict(model,X,Y,t)

U = U.detach().cpu().numpy()
V = V.detach().cpu().numpy()
p = p.detach().cpu().numpy()

U = U.reshape(100,100)
V = V.reshape(100,100)
p = p.reshape(100,100)

U_true = -torch.cos(X)*torch.sin(Y)*torch.cos(t)
U_true = U_true.detach().cpu().numpy()
U_true = U_true.reshape(100,100)
V_true =  torch.sin(X)*torch.cos(Y)*torch.cos(t)
V_true = V_true.detach().cpu().numpy()
V_true = V_true.reshape(100,100)
p_true = -0.25*torch.cos(t)*torch.cos(t)*(torch.cos(2*X)+torch.cos(2*Y))
p_true = p_true.detach().cpu().numpy()
p_true = p_true.reshape(100,100)

U_error = np.mean((U-U_true)**2)/np.mean(U_true**2)
V_error = np.mean((V-V_true)**2)/np.mean(V_true**2)
p_error = np.mean((p-p_true)**2)/np.mean(p_true**2)
print(U_error)
print(V_error)
print(p_error)


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.contourf(x.cpu(), y.cpu(), U, levels=50, cmap="jet")
plt.colorbar()
plt.title(f"PINN Solution (U) to 2D Single Vortex at t={t[0].item():.3f}, RMSE={U_error:.3e}")
plt.xlabel("x")
plt.ylabel("y")
plt.figure(figsize=(8, 6))
plt.contourf(x.cpu(), y.cpu(), V, levels=50, cmap="jet")
plt.colorbar()
plt.title(f"PINN Solution (V) to 2D Single Vortex at t={t[0].item():.3f}, RMSE={V_error:.3e}")
plt.xlabel("x")
plt.ylabel("y")
plt.figure(figsize=(8, 6))
plt.contourf(x.cpu(), y.cpu(), p, levels=50, cmap="jet")
plt.colorbar()
plt.title(f"PINN Solution (p) to 2D Single Vortex at t={t[0].item():.3f}, RMSE={p_error:.3e}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Training
# Hyperparameters
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning_rate = 0.001
# epochs = 50000

# # Create training data
# N_domain   = 1000  # Number of collocation points
# N_boundary = 400  # Number of boundary points
# N_initial  = 100  # Number of initial points

# # Sample points inside the domain
# a,b = -torch.pi/2, torch.pi/2
# x = (b-a)*torch.rand(N_domain, 1, device=device, requires_grad=True)+a
# y = (b-a)*torch.rand(N_domain, 1, device=device, requires_grad=True)+a
# t = (b-a)*torch.rand(N_domain, 1, device=device, requires_grad=True)

# # Boundary points
# x_left     =     a*torch.ones(N_boundary//4, 1, device=device, requires_grad=True)
# y_left     = (b-a)*torch.rand(N_boundary//4, 1, device=device, requires_grad=True)+a
# t_left     = (b-a)*torch.rand(N_boundary//4, 1, device=device, requires_grad=True)
# x_right    =     b*torch.ones(N_boundary//4, 1, device=device, requires_grad=True)
# y_right    = (b-a)*torch.rand(N_boundary//4, 1, device=device, requires_grad=True)+a
# t_right    = (b-a)*torch.rand(N_boundary//4, 1, device=device, requires_grad=True)
# x_bottom   = (b-a)*torch.rand(N_boundary//4, 1, device=device, requires_grad=True)+a
# y_bottom   =     a*torch.ones(N_boundary//4, 1, device=device, requires_grad=True)
# t_bottom   = (b-a)*torch.rand(N_boundary//4, 1, device=device, requires_grad=True)
# x_top      = (b-a)*torch.rand(N_boundary//4, 1, device=device, requires_grad=True)+a
# y_top      =     b*torch.ones(N_boundary//4, 1, device=device, requires_grad=True)
# t_top      = (b-a)*torch.rand(N_boundary//4, 1, device=device, requires_grad=True)

# x_boundary = torch.cat((x_left,x_right,x_bottom,x_top),dim=0)
# y_boundary = torch.cat((y_left,y_right,y_bottom,y_top),dim=0)
# t_boundary = torch.cat((t_left,t_right,t_bottom,t_top),dim=0)

# # Initial condition points
# x_initial = (b-a)*torch.rand(N_initial, 1, device=device, requires_grad=True)+a
# y_initial = (b-a)*torch.rand(N_initial, 1, device=device, requires_grad=True)+a
# t_initial = torch.zeros(N_initial, 1, device=device, requires_grad=True)

# # Model, optimizer, and training
# model = PINN()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(epochs):
#     optimizer.zero_grad()

#     # Compute losses
#     f_loss_u, f_loss_v, f_loss_p = physics_loss(model, x, y, t)
#     b_loss_u, b_loss_v, b_loss_p = boundary_loss(model, x_boundary, y_boundary, t_boundary)
#     i_loss_u, i_loss_v, i_loss_p = initial_loss(model, x_initial, y_initial, t_initial)

#     # Total loss
#     loss_PDE = torch.mean(f_loss_u**2)+torch.mean(f_loss_v**2)+torch.mean(f_loss_p**2)
#     loss_boundary = torch.mean(b_loss_u**2)+torch.mean(b_loss_v**2)+torch.mean(b_loss_p**2)
#     loss_initial  = torch.mean(i_loss_u**2)+torch.mean(i_loss_v**2)+torch.mean(i_loss_p**2)
#     loss = loss_PDE+loss_boundary+loss_initial

#     # Backpropagation
#     loss.backward(retain_graph=True)
#     optimizer.step()

#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: L = {loss.item():.4e},L_PDE = {loss_PDE.item():.4e},L_b = {loss_boundary.item():.4e},L_i = {loss_initial.item():.4e}")

# # Save the trained model
# torch.save(model.state_dict(), "single_vortex.pt")
