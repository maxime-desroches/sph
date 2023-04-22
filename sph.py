#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

N = 300
D = 2
h = 0.1
rho_0 = 3
k = 0.00008
k_near = 0.0008
beta = 0.25

pos = np.random.uniform(low=0.5, high=1.5, size=(N,D))
pos[:,1] = np.random.uniform(low=0, high=1, size=(N,))
pos_prev = np.empty((N,D))
vel = np.random.uniform(low=0.0, high=0.0, size=(N,D))
rho = np.zeros((N))
rho_near = np.zeros((N))
P = np.zeros((N))
P_near = np.zeros((N))
force = np.zeros((N,D))

def normalize(v):
  norm = np.reshape(np.linalg.norm(v,axis=2),(N,N,1))
  norm[norm == 0] = 1
  return v/norm

def physics(pos, vel, rho, rho_near, P, P_near, force):
  #precompute all the N^2 stuff
  r = pos.reshape((N,1,D)) - pos
  rhat = normalize(r)
  q = np.linalg.norm(r, axis=2)
  q[abs(q - h/2) >= h/2] = h
  q = q/h
  Q = 1-q
  Q2 = Q*Q
  Q3 = Q2*Q

  #density
  rho += np.sum(Q2, axis=1)
  rho_near += np.sum(Q3, axis=1)
    
  #pressure
  P = k*(rho - rho_0)
  P_near = k_near*rho_near
  
  #pressure force
  f_vec = ((P.reshape((N,1))+P)*Q2 + (P_near.reshape((N,1))+P_near)*Q3).reshape((N,N,1))*(rhat)
  force += np.sum(f_vec, axis=1)

  #viscosity
  u = np.sum((vel.reshape((N,1,D)) - vel)*(rhat), axis=2)
  u = np.clip(u, 0, None)
  I = (Q*(beta*u)).reshape((N,N,1))*(rhat)
  I = 0.5*I*(np.tri(N, N, k=-1).T.reshape((N,N,1)))
  vel -= np.sum(I, axis=1)
  vel += np.sum(np.stack(I, axis=1), axis=1)

fig = plt.figure()
ax = plt.axes(xlim=[0,1], ylim=[0,0.5])
line, = ax.plot([],[],'co', ms=30)

def update(frame):
  global pos_prev, vel, pos, force, rho, rho_near
  
  #step
  pos_prev = pos
  vel += force
  pos += vel
  rho *= 0
  rho_near *= 0
  vel += pos - pos_prev

  #external force (gravity)
  force = np.array([[0, -0.0045]]*N)

  #bounds for walls
  force -= np.dstack((0.05*(np.clip(pos[:,0], 1, None) - 1) + 0.05*np.clip(pos[:,0], None, 0), 0.05*(np.where(pos[:,1] < 0, pos[:,1], 0.5) - 0.5)))[0]

  #physics
  physics(pos, vel, rho, rho_near, P, P_near, force)

  #render
  line.set_data(pos[:,:1],pos[:,1:])
  return line,

ani = animation.FuncAnimation(fig, update, interval=16, blit=True)
plt.show()
