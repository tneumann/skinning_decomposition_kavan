from __future__ import print_function

from collections import defaultdict
from heapq import heappush, heappop
import plac
import numpy as np
from scipy import weave
from scipy import linalg
from scipy.optimize import nnls

from cgtools import vector as V
from cgtools.io.hdf5 import load_mesh_animation, save_mesh_animation



def initialize_bones(verts, verts0, tris, num_bones):
    """ 
    Implements the algorithm from section 4.1: Initialization 
    """
    num_frames, num_verts, _ = verts.shape
    # find initial cluster centers on rest-shape geometry 
    # use the greedy algorithm for the p-center problem
    pts = verts0[tris].mean(axis=1)
    center_indices = []
    dist_to_nearest_center = np.full(len(pts), np.inf)
    while len(center_indices) < num_bones:
        if len(center_indices) == 0:
            # Kavan doesn't give any detail on which triangle to select
            # for the center of the first proxy bone that is placed
            # I decided to take the one whose mid-point moves the most
            # during the whole animation
            i = verts[:, tris].mean(axis=2).std(axis=0).sum(axis=1).argmax()
        else:
            i = np.argmax(dist_to_nearest_center)
        c = pts[i]
        d = ((pts - c)**2).sum(axis=-1)
        dist_to_nearest_center = np.minimum(d, dist_to_nearest_center)
        center_indices.append(i)

    # construct an adjacency list for vertex neighborhood
    vert_adjacency = defaultdict(list)
    for i1, i2, i3 in tris:
        vert_adjacency[i1].append(i2)
        vert_adjacency[i2].append(i1)
        vert_adjacency[i1].append(i3)
        vert_adjacency[i3].append(i1)
        vert_adjacency[i2].append(i3)
        vert_adjacency[i3].append(i2)

    # for each cluster center, prepare the priority queue 
    prio_queues = [[] for i in xrange(num_bones)]

    # assign vertices of center triangles
    unassigned_vertex_indices = set(range(len(verts0)))
    vertex_clusters = [[] for i in xrange(num_bones)]
    for i, tri_index in enumerate(center_indices):
        for vi in tris[tri_index]: # every vertex in the triangle
            vertex_clusters[i].append(vi)
            if vi in unassigned_vertex_indices:
                unassigned_vertex_indices.remove(vi)

    # calculate deformation gradient for each cluster center and each frame
    D_center = []
    for i, tri_index in enumerate(center_indices):
        u = verts[:, tris[tri_index, 1]] - verts[:, tris[tri_index, 0]]
        v = verts[:, tris[tri_index, 2]] - verts[:, tris[tri_index, 0]]
        n = V.normalized(np.cross(u, v))
        T0_inv = np.linalg.inv(np.column_stack((u[0], v[0], n[0])))
        D = [np.dot(np.column_stack((u[i], v[i], n[i])), T0_inv)
             for i in xrange(num_frames)]
        D_center.append(D)

    D_center = np.array(D_center)
    t_center = verts[:, tris[center_indices, 0]].swapaxes(0, 1)

    def _defgradient_prediction_error(center_index, vert_index):
        e0 = verts[0, vert_index] - t_center[center_index, 0]
        if 0:
            # slow, python-only variant
            verts_predicted = [np.dot(D, e0) + t
                               for D, t in zip(D_center[center_index],
                                               t_center[center_index])]
            return ((np.array(verts_predicted) - verts[:,vert_index])**2).sum()
        else:
            # faster C implementation
            Dc = D_center[center_index]
            vc = verts[:, vert_index]
            ts = t_center[center_index]
            n = len(Dc)
            err = weave.inline("""
                double err = 0.0;
                for(int i=0; i < n; i++) {
                   double px = Dc(i,0,0) * e0(0) + Dc(i,0,1) * e0(1) + Dc(i,0,2) * e0(2);
                   double py = Dc(i,1,0) * e0(0) + Dc(i,1,1) * e0(1) + Dc(i,1,2) * e0(2);
                   double pz = Dc(i,2,0) * e0(0) + Dc(i,2,1) * e0(1) + Dc(i,2,2) * e0(2);
                   double dx = (px - vc(i,0) + ts(i,0));
                   double dy = (py - vc(i,1) + ts(i,1));
                   double dz = (pz - vc(i,2) + ts(i,2));
                   err += dx*dx + dy*dy + dz*dz;
                }
                return_val = err;
                """,
                ['n', 'Dc', 'vc', 'e0', 'ts'], 
                type_converters=weave.converters.blitz)
            return err

    # add vertices of center triangles to the priority queues
    for ci in xrange(num_bones):
        vi_to_add = set()
        for vi_in_cluster in vertex_clusters[ci]:
            for vi in vert_adjacency[vi_in_cluster]:
                if vi in unassigned_vertex_indices:
                    vi_to_add.add(vi)
        for vi in vi_to_add:
            err = _defgradient_prediction_error(ci, vi)
            heappush(prio_queues[ci], (err, vi))

    # process priority queues of each cluster until all of them are empty
    while any([len(q) > 0 for q in prio_queues]):
        # find the vertex giving the lowest 
        # deformation gradient prediction error 
        # and add it to the corresponding cluster center
        best_ci = np.argmin(
            [prio_queues[ci][0][0] if len(prio_queues[ci]) > 0 else np.inf
             for ci in xrange(num_bones)])
        q = prio_queues[best_ci]
        _, vi = heappop(q)
        if vi not in unassigned_vertex_indices:
            # has already been assigned (by another cluster)
            continue
        unassigned_vertex_indices.remove(vi)
        vertex_clusters[best_ci].append(vi)
        # add vertices neighboring the added vertex to the priority queue
        q_current_verts = set(vi for err, vi in q)
        for vi in vert_adjacency[vi]:
            if vi in unassigned_vertex_indices and vi not in q_current_verts:
                err = _defgradient_prediction_error(best_ci, vi)
                heappush(q, (err, vi))

    # convert to vertex labeling
    vertex_label = np.empty(num_verts, np.int)
    vertex_label[:] = -1
    for ci, vertex_indices in enumerate(vertex_clusters):
        vertex_label[vertex_indices] = ci
    assert not np.any(vertex_label == -1), \
            "not all vertices have been labeled - non-manifold geometry?"

    W = np.zeros((num_bones, num_verts))
    W[vertex_label, np.arange(num_verts)] = 1.0

    return center_indices, vertex_label, W

def reduce_dim(X):
    # TODO replace with dimensionality reduction from Kavan's paper
    # this is not the original greedy implementation, but uses truncated svd
    U, s, Vt = linalg.svd(X, full_matrices=False)
    eps = 0.0005 * np.sqrt(X.shape[0] * X.shape[1])
    for d in xrange(1, X.shape[0]):
        B = U[:,:d]
        C = s[:d,np.newaxis] * Vt[:d,:]
        Xrec = np.dot(B, C)
        error = np.linalg.norm(Xrec - X)
        if error < eps:
            print("dimensionality reduced from %d to %d (error %f < eps %f)" % (X.shape[0], d, error, eps))
            break
    return B, C

def form_X_matrix(verts0, W):
    num_bones, num_verts = W.shape
    # the next index line puts the vertices exactly into
    # the matrix format as matrix X in Kavan's paper
    return (V.hom4(verts0).T[np.newaxis,:,:] * W[:,np.newaxis,:])\
            .reshape(4*num_bones, num_verts) # R^(4P \times N)

def optimize_transformations(verts0, W, C):
    X = form_X_matrix(verts0, W)
    try:
        Tr = linalg.solve(np.dot(X, X.T), np.dot(C, X.T).T).T
    except linalg.LinAlgError:
        print("singular matrix in optimize_transformations, using slower pseudo-inverse")
        Tr = np.dot(
            np.linalg.pinv(np.dot(X, X.T)),
            np.dot(C, X.T).T).T
    return Tr
    #T = np.dot(B, Tr)
    #T2 = linalg.solve(np.dot(X, X.T), np.dot(A, X.T).T).T

def optimize_restpose(Tr, W, C):
    num_bones, num_verts = W.shape
    # reshape Tr according to section 4.3 into 
    # a matrix of d * P of 4-vectors
    Tr1 = Tr.reshape((-1, num_bones, 4))
    new_verts0 = []
    for j in xrange(num_verts):
        # multiply with weights and sum over num_bones axis
        Lambda = (Tr1 * W[np.newaxis,:,j,np.newaxis]).sum(axis=1)
        # convert to system matrix - last column must be one, 
        # so subtract from rhs
        gamma = C[:,j] - Lambda[:,3]
        Lambda = Lambda[:,:3]
        v = linalg.solve(np.dot(Lambda.T, Lambda), 
                         np.dot(Lambda.T, gamma))
        new_verts0.append(v)
    return np.array(new_verts0)

def optimize_weights(num_bones, Tr, C, verts0, K, nnls_soft_constr_weight=1.e+4):
    """ optimize for blending weights according to section 4.4 """
    # reshape Tr according to section 4.3 into 
    # a matrix of d * P of 4-vectors
    # TODO: not sure if this is correct already
    #       during iteration this step sometimes increases the residual :-(
    num_verts = verts0.shape[0]
    Tr1 = Tr.reshape((-1, num_bones, 4))
    new_W = np.zeros((num_bones, num_verts))
    for i in xrange(num_verts):
        # form complete system matrix Sigma * w = y
        Sigma = np.inner(Tr1, V.hom4(verts0[i]))
        #import ipdb; ipdb.set_trace()
        y = C[:,i]
        # choose K columns that individually best explain the residual
        error = ( (Sigma - y[:,np.newaxis]) ** 2 ).sum(axis=0)
        k_best = np.argsort(error)[:K]
        # solve nonlinear least squares problem
        # with additional convexity constraint sum(k_weighs) = 1
        Sigma = Sigma[:,k_best]
        #k_weights0 = W[k_best, i]
        k_weights, _ = nnls(
            np.vstack((Sigma, nnls_soft_constr_weight * np.ones(K))),
            np.append(y, nnls_soft_constr_weight))
        new_W[k_best, i] = k_weights
    return new_W


def reconstruct_animation(B, Tr, W, verts0):
    num_verts, num_frames = W.shape
    X = form_X_matrix(verts0, W)
    T = Tr if B is None else np.dot(B, Tr)
    # pure animation matrix
    Arec = np.dot(T, X)
    # reshaped as vertex array
    verts_rec = np.rollaxis(np.rollaxis(Arec.reshape(3, -1, Arec.shape[-1]), 2), 2)
    return Arec, verts_rec

def residual(A, B, C, Tr, W, verts0):
    num_verts, num_frames = W.shape
    Arec, _ = reconstruct_animation(B, Tr, W, verts0)
    return 1000 * np.linalg.norm(Arec - A) / np.sqrt(3 * num_verts * num_frames)


@plac.annotations(
    verbose = ('verbose output during optimization', 'flag', 'v'),
    num_bones = ('number of proxy bones to place', 'positional', None, int),
    num_it = ('number of iterations to perform', 'option', 'i', int),
    K = ('number of bones influencing one vertex', 'option', 'K', int),
    full = ('do not reduce dimensionality before fitting bones', 'flag', 'f'),
)
def main(input_file, output_file, num_bones, K=4, verbose=False, num_it=15, full=False):
    verts, tris = load_mesh_animation(input_file)
    A = np.vstack((verts[:,:,0], verts[:,:,1], verts[:,:,2]))
    verts0 = verts[0]
    num_frames, num_verts, _ = verts.shape

    if num_bones < K:
        raise ValueError("num_bones (=%d) should not be smaller then K (=%d). please pass -K %d or increase the number of bones" % (num_bones, K, num_bones))

    if full:
        B, C = np.eye(A.shape[0]), A
    else:
        print("reducing dimensionality")
        B, C = reduce_dim(A)
    print("place initial bones clusters")
    center_indices, vertex_label, W = initialize_bones(
        verts, verts0, tris, num_bones)

    for it in xrange(num_it):
        Tr = optimize_transformations(verts0, W, C)
        print("iteration % 3d: %f" % (it, residual(A, B, C, Tr, W, verts0)))
        if verbose:
            print("optimize transformations", residual(A, B, C, Tr, W, verts0))
        W = optimize_weights(num_bones, Tr, C, verts0, K)
        if verbose:
            print("optimize weights", residual(A, B, C, Tr, W, verts0))
        verts0 = optimize_restpose(Tr, W, C)
        if verbose:
            print("optimize restpose", residual(A, B, C, Tr, W, verts0))

    Tr = optimize_transformations(verts0, W, C)
    print("residual after optimization", residual(A, B, C, Tr, W, verts0))

    Arec, verts_rec = reconstruct_animation(B, Tr, W, verts0)
    T_full = np.dot(B, Tr).reshape(3, num_frames, num_bones, 4).swapaxes(0, 2).swapaxes(0, 1)

    save_mesh_animation(output_file, verts_rec, tris, 
                        bone_transformations=T_full,
                        bone_blendweights=W, verts_restpose=verts0)


if __name__ == '__main__':
    plac.call(main)

