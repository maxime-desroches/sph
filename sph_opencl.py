#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np
import time

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

N = 1000 
D = 2
S = 1
M = 1
h = 0.1
rho_0 = 3
k = 0.00008
k_near = 0.0008
dt = 1


def normalize(v):
  norm = np.reshape(np.linalg.norm(v,axis=2),(N,N,1))
  norm[norm == 0] = 1
  return v/norm

pos = np.random.uniform(low=0.5, high=1.5, size=(N,D))
pos[:,1] = np.random.uniform(low=0, high=1, size=(N,))
pos_prev = np.empty((N,D))
vel = np.random.uniform(low=0.0, high=1.0, size=(N,D))
rho = np.zeros((N))
rho_near = np.zeros((N))
P = np.zeros((N))
P_near = np.zeros((N))
force = np.zeros((N,D), dtype=np.float64)

program = cl.Program(ctx,''.join(open('kernel.cl','r', encoding='utf-8').readlines())).build()

def physics(pos, vel, rho, rho_near, P, P_near, force):
  global cl, queue, ctx

  st = time.monotonic()
  a = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos)
  b = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, N*N*8)
  c = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, N*N*8)
  d = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, N*N)
  rho_cl = np.zeros((N,N), dtype=np.float64)
  rho_near_cl = np.zeros((N,N), dtype=np.float64)

  program.density(queue, (1000,1000), None, a, b, c).wait()
  cl.enqueue_copy(queue, rho_cl, b).wait()
  cl.enqueue_copy(queue, rho_near_cl, c).wait()

  rho = np.sum(rho_cl, axis=1)
  rho_near = np.sum(rho_near_cl, axis=1)
  print(f"KERNEL DENSITY: {(time.monotonic() - st)}")
  #assert(np.allclose(d_array, rho))
  #assert(np.allclose(dn_array, rho_near))

  st = time.monotonic()
  P = k*(rho - rho_0)
  P_near = k_near*rho_near
  print(f"PRESSURE: {(time.monotonic() - st)}")


  st = time.monotonic()
  pos_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos)
  p_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=P)
  p_near_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=P_near)
  f_cl = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, N*N*D*8)
  f_out = np.zeros((N,N,D), dtype=np.float64)

  program.pressure(queue, (1000,1000), None, pos_cl, p_cl, p_near_cl, f_cl).wait()
  cl.enqueue_copy(queue, f_out, f_cl).wait()

  force += np.sum(f_out, axis=1)
  print(f"KERNEL PRESSURE FORCE: {(time.monotonic() - st)}")

  #assert(np.allclose(f_out, force))


  st = time.monotonic()
  #:(
  print(f"KERNEL VISCOSITY: {(time.monotonic() - st)}")



fig = plt.figure()
ax = plt.axes(xlim=[0,1], ylim=[0,0.5])
line, = ax.plot([],[],'c.', ms=10)

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
