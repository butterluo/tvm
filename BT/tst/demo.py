import tvm
import os
import numpy as np

n = 1024

A = tvm.te.placeholder((n,), name='A')
B = tvm.te.placeholder((n,), name='B')
C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = tvm.te.create_schedule(C.op)
# outer, inner = s[C].split(C.op.axis[0], factor=64)
# s[C].parallel(outer)

tgt = tvm.target.Target(target="llvm", host="llvm")

fadd = tvm.build(s, [A, B, C], tgt, name="vecadd")

dev = tvm.device(tgt.kind.name, 0)
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
print("======= Result Is:")
print(c)
