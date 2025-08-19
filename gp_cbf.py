# Allow Crtl-C to Works Despite Plots
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from tqdm import tqdm
import math as mt
import numpy as np
from numpy import linalg as la
from gp_lib.gp import GaussianProcess as gp
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

class Parameters:
  def __init__(self):
    # Initialize Random Seed
    # np.random.seed(9)

    # Simulation Params
    self.goal = np.array([1.0, 1.0])
    self.init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    self.bbx = np.array([[-0.2, -0.2], [1.2, 1.2]])
    self.use_bbx = True

    # Test With Fixed Obstacles
    self.obs = np.array([[0.00, 0.70, 0.15],
                         [0.33, 0.20, 0.15],
                         [0.66, 0.80, 0.15],
                         [1.00, 0.30, 0.15]])

    # Test With Random Obstacles
    # self.obs = np.array([[np.random.rand(), np.random.rand(), 0.15],
    #                      [np.random.rand(), np.random.rand(), 0.15],
    #                      [np.random.rand(), np.random.rand(), 0.15],
    #                      [np.random.rand(), np.random.rand(), 0.15]])

    self.dif = 0.0001 # Numerical Diff. Precision
    self.dt = 0.001
    self.tf = 40

    self.xx = self.init.reshape((6,1))

    # Lidar Params
    self.sensor_res_degree = 0.5
    self.sensor_range = 0.2

    # Stabiliser Params
    self.kp = 0.2
    self.kv = 0.8
    self.u_max = 2.0
    self.u_min = -2.0

    # CBF Params
    alpha_1 = 0.5
    alpha_2 = 0.5
    self.k = np.array([alpha_1*alpha_2, alpha_1+alpha_2])
    
    # GP Params
    self.nn_samples = 30
    self.h_reg = gp(2, self.nn_samples) # Regress Over x-y
    self.h_reg.set_hyperparams(np.array([0.2, 0.2])) # Best Initial Condition

    self.Lfh_reg = gp(4, self.nn_samples) # Regress Over x-y-vx-vy
    self.Lfh_reg.set_hyperparams(np.array([0.3, 0.3, 0.3, 0.3])) # Best Initial Condition

    self.use_var = False # True for min_vv - False for min_dd
    self.use_cost_var = True

    self.min_vv = 0.001
    self.min_dd = 0.03

    # Only for GPCBF
    self.kk_var = 0.008

    # HGO Params
    self.ll = 5
    self.dd = np.array([10, 50])
    
    # Select The Wanted Simulation
    self.cbf_type = 'GPHGCBF' # NONE - CBF - GPCBF - GPHGCBF
    self.hh_type = 'PROD_DIST' # MIN_DIST - PROD_DIST

    self.save_fig = False
    self.fig_name = 'cbf.png'

def f(x, u):
  xd = np.empty((4))

  # Single Integrator - 2D
  xd[0] = x[2]
  xd[1] = x[3]
  xd[2] = u[0]
  xd[3] = u[1]
  return xd

def obs(params, x, u, y):
  xd = np.empty((2))

  xd[0] = x[5] + (params.ll)*params.dd[0]*(y - x[4])
  xd[1] = (params.ll**2)*params.dd[1]*(y - x[4])

  if params.cbf_type == 'GPHGCBF':
    ff = np.array([[x[2]], [x[3]], [0.0], [0.0]])
    gg = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dLfh_ = params.Lfh_reg.posterior_dmean(x[0:4])

    Lff_h = dLfh_@ff
    Lgf_h = dLfh_@gg

    xd[1] = xd[1] + Lff_h + Lgf_h@u.T
  
  return xd

def h(params, x):
  dd_m_ = mt.inf
  dd_p_ = 1.0

  # Walls
  if params.use_bbx:
    for ii in range(0, params.bbx.shape[0]):
      dx = (x[0] - params.bbx[ii, 0])**2
      dd_p_ = dd_p_*dx
      if dx < dd_m_:
        dd_m_ = dx

      dy = (x[1] - params.bbx[ii, 1])**2
      dd_p_ = dd_p_*dy
      if dy < dd_m_:
        dd_m_ = dy

  # Obstacles
  for ii in range(0, params.obs.shape[0]):
    da = la.norm(x[0:2] - params.obs[ii, 0:2])**2 - params.obs[ii, 2]**2

    dd_p_ = dd_p_*da
    if da < dd_m_:
      dd_m_ = da
      
  if params.hh_type == 'MIN_DIST':
    return dd_m_
  elif params.hh_type == 'PROD_DIST':
    return dd_p_
  else:
    print('Not Implemented')
    return 0.0

def dh(params, x):
  dh = np.empty((4))
  dh[0] = (h(params, x + np.array([params.dif, 0.0])) - h(params, x))/params.dif
  dh[1] = (h(params, x + np.array([0.0, params.dif])) - h(params, x))/params.dif
  dh[2] = 0.0
  dh[3] = 0.0
  return dh

def ddh(params, x):
  ddh = np.empty((4,4))

  dh_ = dh(params, x)
  dh_x_ = dh(params, x + np.array([params.dif, 0.0]))
  dh_y_ = dh(params, x + np.array([0.0, params.dif]))

  ddh[:,0] = (dh_x_ - dh_)/params.dif
  ddh[:,1] = (dh_y_ - dh_)/params.dif
  ddh[:,2] = np.array([0.0, 0.0, 0.0, 0.0])
  ddh[:,3] = np.array([0.0, 0.0, 0.0, 0.0])
  return ddh

def kk(params, x):
  return sat(params, params.kp*(params.goal - x[0:2]) - params.kv*x[2:4])
  
def sat(params, uu):
  for ii in range(0, uu.shape[0]):
    if uu[ii] > params.u_max:
      uu[ii] = params.u_max
    if uu[ii] < params.u_min:
      uu[ii] = params.u_min
  return uu

def setup_data(params, xx):
  # h
  if params.cbf_type == 'CBF':
    params.h_ = h(params, xx[0:2])
  elif params.cbf_type == 'GPCBF':
    params.h_ = params.h_reg.posterior_mean(xx[0:2]) - params.kk_var*params.h_reg.posterior_variance(xx[0:2])
  elif params.cbf_type == 'GPHGCBF':
    params.h_ = xx[4]
  else:
    print('Not Implemented Yet')

  # L_f h
  params.ff = np.array([[xx[2]], [xx[3]], [0.0], [0.0]])
  if params.cbf_type == 'CBF':
    dh_ = dh(params, xx[0:2])
    params.Lf_h = dh_@params.ff
  elif params.cbf_type == 'GPCBF':
    dh_ = params.h_reg.posterior_dmean(xx[0:2]) - params.kk_var*params.h_reg.posterior_dvariance(xx[0:2])
    dh_ = np.concatenate((dh_, np.zeros((1,2))), axis=1)
    params.Lf_h = dh_@params.ff
  elif params.cbf_type == 'GPHGCBF':
    params.Lf_h = xx[5]
  else:
    print('Not Implemented Yet')

  # L_f^2 h
  df = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
  if params.cbf_type == 'CBF':
    ddh_ = ddh(params, xx[0:2])
    params.Lff_h = (params.ff.T@ddh_ + dh_@df)@params.ff
  elif params.cbf_type == 'GPCBF':
    ddh_ = params.h_reg.posterior_ddmean(xx[0:2]) - params.kk_var*params.h_reg.posterior_ddvariance(xx[0:2])
    ddh_ = np.concatenate((ddh_, np.zeros((2,2))), axis=1)
    ddh_ = np.concatenate((ddh_, np.zeros((2,4))), axis=0)
    params.Lff_h = (params.ff.T@ddh_ + dh_@df)@params.ff
  elif params.cbf_type == 'GPHGCBF':
    dLfh_ = params.Lfh_reg.posterior_dmean(xx[0:4])
    params.Lff_h = dLfh_@params.ff
  else:
    print('Not Implemented Yet')

  # L_g L_f h
  params.gg = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
  if params.cbf_type == 'CBF':
    params.Lgf_h = (params.ff.T@ddh_ + dh_@df)@params.gg
  elif params.cbf_type == 'GPCBF':
    params.Lgf_h = (params.ff.T@ddh_ + dh_@df)@params.gg
  elif params.cbf_type == 'GPHGCBF':
    params.Lgf_h = dLfh_@params.gg
  else:
    print('Not Implemented Yet')

def cbf_loss(x, xx, uu, params):
  if params.cbf_type == 'GPHGCBF' and params.use_cost_var:
    return 0.5*((x[0] - uu[0])**2 + (x[1] - uu[1])**2) + params.Lfh_reg.posterior_dvariance(xx[0:4])@params.gg@x
  else:
    return 0.5*((x[0] - uu[0])**2 + (x[1] - uu[1])**2)

def cbf_cons(x, params):
  return (params.Lff_h + params.Lgf_h@x + params.k[0]*params.h_ + params.k[1]*params.Lf_h)[0,0]

def cbf(params, xx, uu):
  if params.cbf_type != 'CBF':
    check_sampling(params, xx)

  setup_data(params, xx)

  cons = ({'type': 'ineq', 'fun': lambda x: cbf_cons(x, params)})
  uu_cbf = minimize(cbf_loss, uu, args=(xx, uu, params), method='SLSQP', constraints=cons, bounds=Bounds(params.u_min, params.u_max), options={'ftol':1e-15}).x
  return uu_cbf

def check_sampling(params, xx):
  if params.cbf_type == 'GPCBF':
    gp_ = params.h_reg
    ys = h(params, xx[0:2])
    xs = xx[0:2]
  elif params.cbf_type == 'GPHGCBF':
    gp_ = params.Lfh_reg
    ys = xx[5]
    xs = xx[0:4]
  else:
    print('Not Implemented')
    return

  if gp_.get_nsample() == 0:
    # Add Train Point
    gp_.add_sample(xs, ys)
    gp_.train()
    params.last_sampled = xs
  else:
    to_be_sampled = False
    if params.use_var:
      # Get Variance at Point
      ss = gp_.posterior_variance(xs)
      to_be_sampled = ss > params.min_vv
    else:
      to_be_sampled = la.norm(xs - params.last_sampled) > params.min_dd

    if(to_be_sampled):
      # Add Train Point
      gp_.add_sample(xs, ys)
      gp_.train()
      gp_.optimize_hyperparameters()
      params.last_sampled = xs

def plot_all(params):
  tspan = np.linspace(0, params.tf, num=params.xx.shape[1])
  fig_1, axs_1 = plt.subplots()

  if params.use_bbx:
    plt.plot([params.bbx[0,0], params.bbx[1,0]], [params.bbx[0,0], params.bbx[0,0]], color='cornflowerblue')
    plt.plot([params.bbx[1,0], params.bbx[1,0]], [params.bbx[0,1], params.bbx[1,1]], color='cornflowerblue')
    plt.plot([params.bbx[0,0], params.bbx[1,0]], [params.bbx[1,1], params.bbx[1,1]], color='cornflowerblue')
    plt.plot([params.bbx[0,0], params.bbx[0,0]], [params.bbx[0,1], params.bbx[1,1]], color='cornflowerblue')

  for ii in range(0, params.obs.shape[0]):
    circ = plt.Circle((params.obs[ii, 0], params.obs[ii, 1]), params.obs[ii, 2], color='cornflowerblue')
    axs_1.add_artist(circ)

  plt.plot(params.xx[0,:], params.xx[1,:], color='seagreen')
  plt.grid()

  dds = []
  dds_gp = []
  for ii in range(0, params.xx.shape[1]):
    dds.append(h(params, params.xx[0:2, ii]))

    if params.cbf_type[1] == True:
      dds_gp.append(params.h_reg.posterior_mean(params.xx[0:2, ii])[0,0])

  fig_2 = plt.figure()
  plt.plot(tspan, dds)
  plt.grid()

  plt.show()

  # Save Avoidance Image
  if params.save_fig:
    fig_1.savefig(params.fig_name, format='png')

def print_report(params):
  print('Sampled Points h_reg: ', params.h_reg.get_nsample())
  print('Sampled Points Lfh_reg: ', params.Lfh_reg.get_nsample())

# Main #############
params = Parameters()

tspan = np.arange(0, params.tf, params.dt)
for tt in tqdm(tspan):
  if h(params, params.xx[0:2,-1]) < 0.0:
    break

  uu = kk(params, params.xx[:,-1])
  if params.cbf_type != 'NONE':
    uu = cbf(params, params.xx[:,-1], uu)

  xk = np.empty((6))
  xk[0:4] = params.xx[0:4,-1] + params.dt*f(params.xx[:,-1], uu)
  xk[4:6] = params.xx[4:6,-1] + params.dt*obs(params, params.xx[:,-1], uu, h(params, params.xx[0:2,-1]))

  params.xx = np.append(params.xx, xk.reshape((6,1)), axis=1)

print_report(params)
plot_all(params)