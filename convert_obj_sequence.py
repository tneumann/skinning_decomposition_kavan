from __future__ import print_function

from glob import glob
from os import path

import plac

from cgtools.io.obj import load_obj
from cgtools.io.hdf5 import save_mesh_animation


def convert_obj_sequence(directory, hdf_output_file):
    verts_all = []
    tris = None
    files = sorted(glob(path.join(directory, '*.obj')))
    for i, f in enumerate(files):
        print("loading file %d/%d [%s]" % (i+1, len(files), f))
        verts, new_tris = load_obj(f)
        if tris is not None and new_tris.shape != tris.shape and new_tris != tris:
            raise ValueError("inconsistent topology between meshes of different frames")
        tris = new_tris
        verts_all.append(verts)

    save_mesh_animation(hdf_output_file, verts_all, tris)
    print("saved as %s" % hdf_output_file)


if __name__ == "__main__":
    plac.call(convert_obj_sequence)
