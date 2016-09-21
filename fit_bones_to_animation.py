import numpy as np
import plac

from cgtools.io.hdf5 import load_mesh_animation, save_mesh_animation

from kavan import optimize_transformations, reconstruct_animation, residual


def main(input_animation_with_bones, input_animation_to_fit, output_animation):
    _, tris0, verts0, W = load_mesh_animation(input_animation_with_bones, 'verts_restpose', 'bone_blendweights')
    num_bones = W.shape[0]
    verts, tris = load_mesh_animation(input_animation_to_fit)
    num_frames = verts.shape[0]
    A = np.vstack((verts[:,:,0], verts[:,:,1], verts[:,:,2]))
    assert np.all(tris0 == tris)

    T = optimize_transformations(verts0, W, A)
    print "residual", residual(A, None, None, T, W, verts0)
    Arec, verts_rec = reconstruct_animation(None, T, W, verts0)

    T_full = T.reshape(3, num_frames, num_bones, 4).swapaxes(0, 2).swapaxes(0, 1)
    save_mesh_animation(output_animation, verts_rec, tris, 
                        verts_restpose=verts0, bone_blendweights=W, bone_transformations=T_full)


if __name__ == '__main__':
    plac.call(main)

