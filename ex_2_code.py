import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
### Global variables
k = 0.835

def get_N_t_N_x(t_end,x_end,dt,dx):

    N_t = t_end//dt +1
    N_x = x_end//dx +1 

    return N_t,N_x

def initialise_grid(N_t,N_x,Tt_0,Tx_0,Tx_L):

    grid = np.zeros((N_t,N_x))

    #initial values 
    grid[0,:] = Tt_0
    #boundary values 
    grid[:,0] = Tx_0
    grid[:,N_x-1] = Tx_L

    return grid

def update_temp(dx,dt,T_txm,T_tx,T_txp):

    T_tpx = ((k*dt)/(dx**2))*(T_txm + T_tx*(((dx**2)/(k*dt))-2)+T_txp)

    return T_tpx

def update_temp_iter(T_grid,N_t,N_x,dt,dx):

    for j in np.arange(0,N_t-1):
        for i in np.arange(1,N_x-1):
            #print(T_grid[j,i-1],T_grid[j,i],T_grid[j,i+1])

            T_grid[j+1,i] = update_temp(dx,dt,T_grid[j,i-1],T_grid[j,i],T_grid[j,i+1])
    
    return T_grid

def get_ana_T(x,x_end,t,N=10000):

    temperature = 0
    for i in np.arange(1,N,2):

        sin_term = np.sin((i*np.pi*x)/x_end)
        exp_term = np.exp(-((i**2)*(np.pi**2)*k*t/(x_end**2)))

        temperature += (1/i)*sin_term*exp_term

    temperature = (2000/np.pi)*temperature
    return temperature

def get_ana_T_vals_fixed_x(t_vals,x_vals,x_int):

    temp_vals = np.empty((len(t_vals)))
    for i,t in enumerate(t_vals):
        temp_vals[i] = get_ana_T(x_vals[x_int],x_vals[-1],t)
    return temp_vals

def get_Tgrid_ana_T_error(T_grid,N_t,N_x,dt,dx,x_int):

    
    t_vals = np.arange(0,N_t*dt,dt)
    x_vals = np.arange(0,N_x*dx,dx)

    T_grid = update_temp_iter(T_grid,N_t,N_x,dt,dx)
    ana_T = get_ana_T_vals_fixed_x(t_vals,x_vals,x_int)
    error = abs(T_grid[:,x_int]-ana_T)

    return T_grid,ana_T,error

def plot_TxTL_exp(T_grid,t_vals,x_vals):

    fig,ax = plt.subplots(figsize =(8,8))
    for i,temps in enumerate(T_grid):
        plt.plot(x_vals,temps,label="t={}s".format(t_vals[i]))
    plt.xlabel('X Position (cm)')
    plt.ylabel('Temperature (C$\degree$)')
    #plt.legend(loc='center left',bbox_to_anchor=(1,0.5,0.5,0.0))
    plt.legend(fontsize=8.7)
    plt.show()

def plot_Tt_exp(ana_T_vals,T_grid,x_int,t_vals,x_vals):

    T_grid_x_int = T_grid[:,x_int]
    fig,ax = plt.subplots(figsize =(8,8))
    plt.plot(t_vals,T_grid_x_int,label='Explicit Numerical Solution')
    plt.plot(t_vals,ana_T_vals,label='Analytical Solution')
    plt.plot(t_vals,abs(T_grid_x_int-ana_T_vals),label='Absolute Numerical Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (X = {} cm) (C$\degree$)'.format(x_vals[x_int]))
    plt.legend(loc='center left',bbox_to_anchor=(1,0.5,0.5,0.0))
    plt.show()

def init_cn_matrix(N_x,dt,dx):

    cn_matrix = np.zeros((N_x-2,N_x-2))
    gamma = (k*dt)/(2*(dx)**2)
    
    for i in np.arange(N_x-2):

        cn_matrix[i,i] = (1+2*gamma)

        if i < (N_x-3):
            cn_matrix[i,i+1] = -gamma

        if i > 0:
            cn_matrix[i,i-1] = -gamma

    return cn_matrix

def Thomas_algorithm(cn_matrix,rhs_vec):

    cn_shape = np.shape(cn_matrix)
    decomposed_matrix = np.zeros(cn_shape)

    decomposed_matrix[0,:] = cn_matrix[0,:]

    betas = np.zeros(cn_shape[1])
    alphas = np.zeros(cn_shape[1]-1)
    betas[0] = cn_matrix[0,0]
    alphas[0] = cn_matrix[1,0]

    forward_vec = np.zeros(cn_shape[1])
    forward_vec[0] = rhs_vec[0]

    for i in np.arange(1,cn_shape[0]):
        #step 1: Decomposition of CN matrix
        alphas[i-1] = cn_matrix[i,i-1]/betas[i-1]
        betas[i] = cn_matrix[i,i] - alphas[i-1]*cn_matrix[i-1,i]

        decomposed_matrix[i,i-1] = alphas[i-1]
        decomposed_matrix[i,i] = betas[i]

        if i < cn_shape[0]-1:
            decomposed_matrix[i,i+1] = cn_matrix[i,i+1]

        forward_vec[i] = rhs_vec[i] - alphas[i-1]*forward_vec[i-1]

    x_vec = np.zeros(cn_shape[1])

    for j in reversed(np.arange(cn_shape[0])):

        if j < cn_shape[0]-1:
            x_vec[j] = (forward_vec[j] - decomposed_matrix[j,j+1]*x_vec[j+1])/betas[j]
        else:
            x_vec[j] = forward_vec[j]/betas[j]

    return x_vec

def init_rhs_vec(T_grid,dt,dx,j):

    gamma = (k*dt)/(2*(dx)**2)
    rhs_vec_len = np.shape(T_grid)[1]-2

    rhs_vec = np.zeros(rhs_vec_len)
    for i,_ in enumerate(rhs_vec):
        if i==0:
            rhs_vec[i] = gamma*(T_grid[j,0]+T_grid[j+1,0]+T_grid[j,2])+(1-2*gamma)*T_grid[j,1]
        elif i == rhs_vec_len-1:
            rhs_vec[i] = gamma*(T_grid[j,i+2]+T_grid[j+1,i+2]+T_grid[j,i])+(1-2*gamma)*T_grid[j,i+1]
        else:
            rhs_vec[i] = gamma*(T_grid[j,i]+T_grid[j,i+2])+(1-2*gamma)*T_grid[j,i+1]

    return rhs_vec

def cn_implicit_solver(T0_grid,N_t,N_x,dt,dx):

    cn_matrix = init_cn_matrix(N_x,dt,dx)

    t_vals = np.arange(0,(N_t+1)*dt,dt)
    T_grid = T0_grid

    for j in np.arange(0,N_t-1):

        rhs_vec = init_rhs_vec(T_grid,dt,dx,j)
        new_T_vals = Thomas_algorithm(cn_matrix,rhs_vec)
        T_grid[j+1,1:N_x-1]  = new_T_vals

    plt.imshow(T_grid, interpolation='none')
    plt.show()
    
    return T_grid

def initial_values_f(x,g=0.1,c=400):

    initial_values = -x*g*(x-x[-1]) + c

    return initial_values

def init_grid_new(N_t,N_x,dx,Tx_0=0,Tx_L=0,g=0.1,c=400):

    grid = np.zeros((N_t,N_x))
    x_vals = np.arange(0,(N_x)*dx,dx)

    #initial values 
    grid[0,:] = initial_values_f(x_vals,g=g,c=c)
    #boundary values 
    grid[:,0] = Tx_0
    grid[:,N_x-1] = Tx_L

    return grid

def sinusoidal(x,a=100,c=200,d=0):

    sinusoidal_val = a*(np.sin(x+d))+c

    return sinusoidal_val

def init_grid_sin_bvp(N_t,N_x,dt,a=100,c=200):

    grid = np.zeros((N_t,N_x))
    t_vals = np.arange(0,(N_t)*dt,dt)
    grid[:,0] = sinusoidal(t_vals)
    grid[:,N_x-1] = sinusoidal(t_vals)

    return grid

def main():

    tt = 600
    xx = 100
    dt = 100
    dx = 5
    T_x0 = 0
    T_xL = 0
    T_t0 = 500

    t_vals = np.arange(0,tt+dt,dt)
    x_vals = np.arange(0,xx+dx,dx)
    N_t,N_x = get_N_t_N_x(tt,xx,dt,dx)

    temp_grid = initialise_grid(N_t,N_x,T_t0,T_x0,T_xL)

    temp_grid,ana_T_vals,abs_error = get_Tgrid_ana_T_error(temp_grid,N_t,N_x,dt,dx,1)
    plot_TxTL_exp(temp_grid,t_vals,x_vals)
    plot_Tt_exp(ana_T_vals,temp_grid,1,t_vals,x_vals)

    CN = init_cn_matrix(N_x,dt,dx)

    temp_grid_early = init_grid_new(N_t,N_x,dx)
    #temp_grid_sin = init_grid_sin_bvp(N_t,N_x,dt)
    #print(x_vals)
    #print(initial_values_f(x_vals))

    temp_grid = initialise_grid(N_t,N_x,T_t0,T_x0,T_xL)
    T_grid = cn_implicit_solver(temp_grid,N_t,N_x,dt,dx)
    plot_TxTL_exp(T_grid,t_vals,x_vals)

main()

#test_matrix = np.reshape(np.array(([2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2])),(4,4))
    #print(test_matrix)
    #test_vector = np.array([4,2,2,10])
    #print(Thomas_algorithm(test_matrix,test_vector))