import numpy as np
import math as mt
from scipy.optimize import minimize
from scipy.optimize import Bounds

class Parameters:
  def __init__(self, n, nn):
    self.order_input = n
    self.nn_sample = nn
    self.gp_L = np.empty((0))
    self.gp_alpha = np.empty((0))
    self.gp_x = np.empty((n,0))
    self.gp_y = np.empty((1,0))
    self.s = 1e-6
    self.dd = 0.0001 # Numerical Diff. Precision
    self.tau = np.ones((n,))

class GaussianProcess:
  def __init__(self, n, nn):
    self.params = Parameters(n, nn)

  def k(self, x1, x2, tau):
    if len(x1.shape) == 1:
      x1 = x1.reshape((self.params.order_input, 1))

    if len(x2.shape) == 1:
      x2 = x2.reshape((self.params.order_input, 1))

    n1 = x1.shape[1]
    n2 = x2.shape[1]
    kk = np.empty((n1, n2))

    for ii in range(0, n1):
      for jj in range(0, n2):
        for idx in range(0, self.params.order_input):
          d = (x1[idx,ii]-x2[idx,jj])**2
          if idx == 0:
            kk[ii,jj] = np.exp(-d/(2*tau[idx]**2))
          else:
            kk[ii,jj] = kk[ii,jj]*np.exp(-d/(2*tau[idx]**2))

    return kk

  def dk(self, x1, x2, tau):
    if len(x1.shape) == 1:
      x1 = x1.reshape((self.params.order_input, 1))

    if len(x2.shape) == 1:
      x2 = x2.reshape((self.params.order_input, 1))

    if x1.shape[1] != 1:
      print('Error')
      return

    nn = x2.shape[1]
    dk = np.empty((self.params.order_input, nn))
    kk = self.k(x1, x2, tau)

    for ii in range(0, nn):
      for idx in range(0, self.params.order_input):
        dk[idx, ii] = -(x1[idx,0] - x2[idx,ii])*kk[0,ii]/(tau[idx]**2)
    return dk

  def ddk(self, x1, x2, tau):
    if len(x1.shape) == 1:
      x1 = x1.reshape((self.params.order_input, 1))

    if len(x2.shape) == 1:
      x2 = x2.reshape((self.params.order_input, 1))

    if x1.shape[1] != 1:
      print('Error')
      return

    nn = x2.shape[1]
    ddk = np.empty((self.params.order_input*2, nn))
    
    kk = self.k(x1, x2, tau)
    dk = self.dk(x1, x2, tau)

    for ii in range(0, nn):
      for idx in range(0, self.params.order_input):
        for idx_ in range(0, self.params.order_input):
          if idx == idx_:
            ddk[idx*self.params.order_input + idx_, ii] = (x2[idx_,ii]**2 - 2*x2[idx_,ii]*x1[idx_,0] + x1[idx_,0]**2 - tau[idx_]**2)*kk[0,ii]/(tau[idx_]**4)
          else:
            ddk[idx*self.params.order_input + idx_, ii] = -(x1[idx_,0] - x2[idx_,ii])*dk[idx_,ii]/(tau[idx_]**2)

    return ddk

  def train(self):
    kk = self.k(self.params.gp_x, self.params.gp_x, self.params.tau)
    self.params.gp_L = np.linalg.cholesky(kk + self.params.s*np.eye(kk.shape[0]))
    self.params.gp_alpha = np.linalg.solve(np.transpose(self.params.gp_L), np.linalg.solve(self.params.gp_L, self.params.gp_y.transpose()))

  def add_sample(self, x, y):
    x = x.reshape((self.params.gp_x.shape[0],1))
    y = y.reshape((self.params.gp_y.shape[0],1))
    self.params.gp_x = np.append(self.params.gp_x, x, axis=1)
    self.params.gp_y = np.append(self.params.gp_y, y, axis=1)

    if(self.params.gp_y.shape[1] > self.params.nn_sample):
      self.params.gp_x = np.delete(self.params.gp_x, 0, 1)
      self.params.gp_y = np.delete(self.params.gp_y, 0, 1)

  def posterior_mean(self, x): 
    x = x.reshape(self.params.gp_x[:,0].shape)
    kk = self.k(x, self.params.gp_x, self.params.tau)
    return kk@self.params.gp_alpha

  def posterior_variance(self, x):
    x = x.reshape(self.params.gp_x[:,0].shape)
    kk = self.k(x, self.params.gp_x, self.params.tau)
    kk_ = self.k(x, x, self.params.tau)
    alpha = np.linalg.solve(np.transpose(self.params.gp_L), np.linalg.solve(self.params.gp_L, np.transpose(kk)))
    return abs(kk_ - kk@alpha)

  def posterior_dmean(self, x):
    x = x.reshape(self.params.gp_x[:,0].shape)
    dk = self.dk(x, self.params.gp_x, self.params.tau)
    return (dk@self.params.gp_alpha).T

  def posterior_ddmean(self, x):
    x = x.reshape(self.params.gp_x[:,0].shape)
    ddk = self.ddk(x, self.params.gp_x, self.params.tau)
    return (ddk@self.params.gp_alpha).reshape((self.params.order_input, self.params.order_input))

  def posterior_dvariance(self, x):
    x = x.reshape(self.params.gp_x[:,0].shape)
    kk = self.k(x, self.params.gp_x, self.params.tau)
    dk = self.dk(x, self.params.gp_x, self.params.tau)
    dk_ = self.dk(x, x, self.params.tau)

    dv = np.empty((self.params.order_input))
    alpha = np.linalg.solve(np.transpose(self.params.gp_L), np.linalg.solve(self.params.gp_L, np.transpose(kk)))
    dv = dk_ - 2*dk@alpha
    return dv.T

  def posterior_ddvariance(self, x):
    x = x.reshape(self.params.gp_x[:,0].shape)
    kk = self.k(x, self.params.gp_x, self.params.tau)
    dk = self.dk(x, self.params.gp_x, self.params.tau)
    ddk = self.ddk(x, self.params.gp_x, self.params.tau)
    ddk_ = self.ddk(x, x, self.params.tau)

    ddv = np.empty((self.params.order_input*2))
    alpha = np.linalg.solve(np.transpose(self.params.gp_L), np.linalg.solve(self.params.gp_L, np.transpose(kk)))
    alpha_ = np.linalg.solve(np.transpose(self.params.gp_L), np.linalg.solve(self.params.gp_L, np.transpose(dk)))

    ddv = (ddk_ - 2*ddk@alpha).reshape((self.params.order_input, self.params.order_input)) - 2*dk@alpha_
    return ddv

  def get_numerical_dx(self, x):
    dx = np.empty((self.params.order_input))
    mm_d_ = np.empty((self.params.order_input))

    mm_ = self.posterior_mean(x)
    for ii in range(0, self.params.order_input):
      dd = np.zeros((self.params.order_input))
      dd[ii] = self.params.dd
      mm_d_[ii] = self.posterior_mean(x + dd) - mm_

    dx = mm_d_/self.params.dd
    return dx

  def get_numerical_dv(self, x):
    dv = np.empty((self.params.order_input))
    vv_d_ = np.empty((self.params.order_input))

    vv_ = self.posterior_variance(x)
    for ii in range(0, self.params.order_input):
      dd = np.zeros((self.params.order_input))
      dd[ii] = self.params.dd
      vv_d_[ii] = self.posterior_variance(x + dd) - vv_

    dv = vv_d_/self.params.dd
    return dv

  def get_numerical_ddx(self, x):
    ddx = np.empty((self.params.order_input, self.params.order_input))
    dx_d_ = np.empty((self.params.order_input, self.params.order_input))

    dx_ = self.get_numerical_dx(x)
    for ii in range(0, self.params.order_input):
      dd = np.zeros((self.params.order_input))
      dd[ii] = self.params.dd

      dx_d_ = self.get_numerical_dx(x + dd)
      ddx[:,ii] = (dx_d_ - dx_)/self.params.dd
    return ddx

  def get_numerical_ddv(self, x):
    ddv = np.empty((self.params.order_input, self.params.order_input))
    dv_d_ = np.empty((self.params.order_input, self.params.order_input))

    dv_ = self.get_numerical_dv(x)
    for ii in range(0, self.params.order_input):
      dd = np.zeros((self.params.order_input))
      dd[ii] = self.params.dd

      dv_d_ = self.get_numerical_dv(x + dd)
      ddv[:,ii] = (dv_d_ - dv_)/self.params.dd
    return ddv

  def optimize_hyperparameters(self):
    self.params.tau = minimize(self.ll, self.params.tau, method='L-BFGS-B', bounds=Bounds(1e-3, mt.inf)).x

  def ll(self, x):
    kk = self.k(self.params.gp_x, self.params.gp_x, np.array(x))
    L = np.linalg.cholesky(kk + self.params.s*np.eye(kk.shape[0]))
    alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, self.params.gp_y.transpose()))
    return 0.5*(self.params.gp_y@alpha)[0,0] + 0.5*mt.log(np.linalg.det(kk + self.params.s*np.eye(kk.shape[0]))) + 0.5*self.get_nsample()*(mt.log(2*mt.pi))

  def get_nsample(self):
    return self.params.gp_x.shape[1]

  def get_hyperparams(self):
    return self.params.tau

  def set_noisepw(self, s):
    self.params.s = s

  def set_hyperparams(self, tau):
    if tau.shape != self.params.tau.shape:
      print('Wrong Shape Hyperparams')
      return
    self.params.tau = tau