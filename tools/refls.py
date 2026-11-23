import struct, sys, numpy as np
from collections import OrderedDict
def read(path):
    out = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if len(h) < 8: break
            step, ln = struct.unpack('<II', h)
            name = f.read(ln).decode(); typ, = struct.unpack('<I', f.read(4))
            ne = struct.unpack('<4q', f.read(32)); n = ne[0]*ne[1]*ne[2]*ne[3]
            data = np.frombuffer(f.read(4*n), dtype=np.int32 if typ else np.float32).reshape(ne[::-1])
            out.append((step, name, data))
    return out
if __name__ == '__main__':
    recs = read(sys.argv[1]); step = int(sys.argv[2])
    names = OrderedDict()
    for s, n, d in recs:
        if s == step:
            base = n.rsplit('-', 1)[0] if n.rsplit('-',1)[-1].isdigit() else n
            names.setdefault(base, []).append((n, d.shape))
    for b, v in names.items(): print(f'{b:32s} x{len(v):3d}  e.g. {v[0][0]} {v[0][1]}')
