import taichi as ti

ti.init(arch=ti.cuda)

x = ti.field(dtype=ti.i16)

ti.root.pointer(ti.i, 1024).dense(ti.i, 1024 * 1024).place(x)
# A sparse array. Each dense block is 2MB in size.

# Populate 1024 * 2MB = 2GB memory
def populate():
  for k in range(1024):
    x[k * 1024 * 1024] = 1

populate()
