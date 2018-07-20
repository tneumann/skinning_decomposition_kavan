import sys
from os import path
import re
import numpy as np
import plac

from cgtools.io.obj import load_obj
from cgtools.io.hdf5 import save_mesh_animation

import kavan


def main(sma_filename, out_hdf5_animation):
    lines = filter(lambda line: not line.startswith('#'),
                   open(sma_filename).readlines())
    W = None
    T = []
    obj_file = None
    num_frames = None

    while len(lines) > 0:
        line = lines.pop(0)
        if line.startswith('*OBJFILENAME'):
            obj_file = path.join(
                path.dirname(sma_filename), 
                lines.pop(0).strip())

        if line.startswith('*BONEANIMATION'):
            m = re.search('BONEINDEX\s*=\s*([\d]+),\s*NFRAMES\s*=\s*([\d]+)', line)
            bone_index, num_frames = map(int, m.groups())
            assert bone_index == len(T)
            curr_ts = []
            for f in xrange(num_frames):
                values = lines.pop(0).strip().split()
                assert int(values[0]) == f
                curr_ts.append(map(float, values[1:]))
            T.append(curr_ts)

        if line.startswith('*VERTEXWEIGHTS'):
            m = re.search('NVERTICES\s*=\s*([\d]+)\s', line)
            n_verts = int(m.groups()[0])
            W = np.zeros((len(T), n_verts))
            for i in xrange(n_verts):
                values = lines.pop(0).strip().split()
                assert int(values[0]) == i
                for bone_index, weight in zip(values[1::2], values[2::2]):
                    W[int(bone_index), i] = float(weight)

    num_bones, num_verts = W.shape
    print("got %d bones" % num_bones)
    # bring T into correct shape
    T = np.array(T)
    # collection of row-major 4x4 matrices, throw away the last column (always 0,0,0,1)
    T = T.reshape(T.shape[0], T.shape[1], 4, 4)[:,:,:3]
    T_proper = T.swapaxes(0, 1).copy()

    # bring rows (x,y,z) to the front
    T = T.swapaxes(0, 2)
    T = T.reshape(3*num_frames, 4*num_bones)

    verts0, tris = load_obj(obj_file)
    Arec, verts = kavan.reconstruct_animation(None, T, W, verts0)

    save_mesh_animation(out_hdf5_animation, verts, tris, 
                        bone_transformations=T_proper, bone_blendweights=W, verts_restpose=verts0)


if __name__ == "__main__":
    plac.call(main)

