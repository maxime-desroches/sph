# SPH fluid simulator

## Dependency
```
python, NumPy, matplotlib
```
## Usage
```
$ python sph.py
```
or change the sha-bang at the top of the file and run it however you like.

A window should then spawn.

## openCL (optional)
if you want to use the kernel, you will need pyopencl. Install it with
```
pip install pyopencl
```
You should check if your harware (GPU or CPU) have implemented openCL. All Nvidia/AMD GPUs  and all Intel CPUs should work. Might work on a mac.
