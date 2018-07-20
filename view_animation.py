#!/usr/bin/env python
import h5py
import numpy as np
from itertools import cycle

import plac
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input, configure_input_data, configure_source_data
from traits.api import HasTraits, Instance, on_trait_change, Enum, Button, Bool
from traitsui.api import View, Group, Item
from pyface.api import FileDialog, OK
from matplotlib.colors import hsv_to_rgb

from cgtools.vis.animator import Animator
from cgtools.skinning import blend_skinning
from cgtools import vector as V
from cgtools.io.off import save_off



def hue_linspace_colors(n, sat=1.0, light=0.5):
    return (hsv_to_rgb(np.dstack((
        np.linspace(0., 1., n, endpoint=False), 
        np.ones(n) * sat,
        np.ones(n) * light)))[0] * 255).astype(np.uint8)

def vertex_weights_to_colors(W, sat=1.0, light=1.0):
    colors = hue_linspace_colors(W.shape[0], sat, light)
    return  (W[:,:,np.newaxis] * colors[:,np.newaxis,:]).sum(axis=0).astype(np.uint8)


def main(*anim_files):
    fig = mlab.figure(bgcolor=(1,1,1))
    all_verts = []
    pds = []
    actors = []
    datasets = []
    glyph_pds = []
    glyph_actors = []

    colors = cycle([(1,1,1), (1,0,0), (0,1,0), (0,0,1)])

    for i, (f, color) in enumerate(zip(anim_files, colors)):
        data = h5py.File(f, 'r')
        verts = data['verts'].value
        tris = data['tris'].value
        print f
        print "  Vertices: ", verts.shape
        print "  Triangles: ", tris.shape
        datasets.append(data)

        # setup mesh
        pd = tvtk.PolyData(points=verts[0], polys=tris)
        normals = tvtk.PolyDataNormals(compute_point_normals=True, splitting=False)
        configure_input_data(normals, pd)
        actor = tvtk.Actor(mapper=tvtk.PolyDataMapper())
        configure_input(actor.mapper, normals)
        actor.mapper.immediate_mode_rendering = True
        actor.visibility = False
        fig.scene.add_actor(actor)

        actors.append(actor)
        all_verts.append(verts)
        pds.append(normals)

        # setup arrows
        arrow = tvtk.ArrowSource(tip_length=0.25, shaft_radius=0.03, shaft_resolution=32, tip_resolution=4)
        glyph_pd = tvtk.PolyData()
        glyph = tvtk.Glyph3D()
        scale_factor = verts.reshape(-1, 3).ptp(0).max() * 0.1
        glyph.set(scale_factor=scale_factor, scale_mode='scale_by_vector', color_mode='color_by_scalar')
        configure_input_data(glyph, glyph_pd)
        configure_source_data(glyph, arrow)
        glyph_actor = tvtk.Actor(mapper=tvtk.PolyDataMapper(), visibility=False)
        configure_input(glyph_actor.mapper, glyph)
        fig.scene.add_actor(glyph_actor)

        glyph_actors.append(glyph_actor)
        glyph_pds.append(glyph_pd)

    actors[0].visibility = True
    glyph_actors[0].visibility = True


    class Viewer(HasTraits):
        animator = Instance(Animator)
        visible = Enum(*range(len(pds)))
        normals = Bool(True)
        export_off = Button
        restpose = Bool(True)
        show_scalars = Bool(True)
        show_bones = Bool(True)
        show_actual_bone_centers = Bool(False)

        def _export_off_changed(self):
            fd = FileDialog(title='Export OFF', action='save as', wildcard='OFF Meshes|*.off')
            if fd.open() == OK:
                v = all_verts[self.visible][self.animator.current_frame]
                save_off(fd.path, v, tris)

        @on_trait_change('visible, normals, restpose, show_scalars, show_bones')
        def _changed(self):
            for a in actors + glyph_actors:
                a.visibility = False
            for d in pds:
                d.compute_point_normals = self.normals
            actors[self.visible].visibility = True
            actors[self.visible].mapper.scalar_visibility = self.show_scalars
            glyph_actors[self.visible].visibility = self.show_bones
            #for i, visible in enumerate(self.visibilities):
            #    actors[i].visibility = visible
            self.animator.render = True

        def show_frame(self, frame):
            v = all_verts[self.visible][frame]
            dataset = datasets[self.visible]
            if not self.restpose:
                rbms_frame = dataset['rbms'][frame]
                v = v * dataset.attrs['scale'] + dataset.attrs['verts_mean']
                v = blend_skinning(
                    v, dataset['segments'].value, rbms_frame,
                    method=dataset.attrs['skinning_method'])
            pds[self.visible].input.points = v
            if 'scalar' in dataset and self.show_scalars:
                if dataset['scalar'].shape[0]  == all_verts[self.visible].shape[0]:
                    scalar = dataset['scalar'][frame]
                else:
                    scalar = dataset['scalar'].value
                pds[self.visible].input.point_data.scalars = scalar
            else:
                pds[self.visible].input.point_data.scalars = None
            if 'bone_transformations' in dataset and self.show_bones:
                W = dataset['bone_blendweights'].value
                T = dataset['bone_transformations'].value
                gpd = glyph_pds[self.visible]
                if self.show_actual_bone_centers:
                    verts0 = dataset['verts_restpose'].value
                    mean_bonepoint = verts0[W.argmax(axis=1)]# - T[0,:,:,3]
                    #mean_bonepoint = np.array([
                    #    np.average(verts0, weights=w, axis=0) for w in W])
                    #gpd.points = np.repeat(mean_bonepoint + T[frame,:,:,3], 3, 0)
                    #print np.tile(mean_bonepoint + T[frame,:,:,3], 3).reshape((-1, 3))
                    #gpd.points = np.tile(mean_bonepoint + T[frame,:,:,3], 3).reshape((-1, 3))
                    pts = []
                    for i in xrange(T.shape[1]):
                        #offset = V.transform(V.hom4(mean_bonepoint[i]), T[frame,i,:,:])
                        offset = np.dot(T[frame, i], V.hom4(mean_bonepoint[i]))
                        pts += [offset] * 3
                    gpd.points = np.array(pts)

                else:
                    bonepoint = np.array([
                        np.average(v, weights=w, axis=0) for w in W])
                    gpd.points = np.repeat(bonepoint, 3, 0)
                gpd.point_data.vectors = \
                        np.array(map(np.linalg.inv, T[frame,:,:,:3])).reshape(-1, 3)
                # color vertices
                vert_colors = vertex_weights_to_colors(W)
                pds[self.visible].input.point_data.scalars = vert_colors
                bone_colors = hue_linspace_colors(W.shape[0], sat=0.8, light=0.7)
                gpd.point_data.scalars = np.repeat(bone_colors, 3, 0)

        view = View( Group(
            Group(
                Item('visible'),
                Item('export_off'),
                Item('normals'),
                Item('restpose'),
                Item('show_scalars'),
                Item('show_bones'),
                Item('show_actual_bone_centers'),
                label="Viewer"
            ),
            Item('animator', style='custom', show_label=False),
            )
        )


    app = Viewer()
    animator = Animator(verts.shape[0], app.show_frame)
    app.animator = animator
    app.edit_traits()
    mlab.show()

if __name__ == "__main__":
    plac.call(main)
