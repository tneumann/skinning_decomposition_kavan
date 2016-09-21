
Python implementation of the paper

        Ladislav Kavan, Peter-Pike Sloan, Carol O'Sullivan.
        "Fast and Efficient Skinning of Animated Meshes."
        Computer Graphics Forum 29(2) [Proceedings of Eurographics], 2010

This is a re-implementation of the algorithm described in Kavan et al.'s paper. The code is in no way associated with any of the authors of the paper. I did the reimplementation for a state-of-the-art comparison for my paper "Sparse Localized Deformation Components" (SIGGRAPH Asia 2013)

## kavan.py

Implements the main algorithm. There is some documentation of the command line parameters:
$ python kavan.py --help

## convert_sma_data.py

Can convert the data from the 
[project page of the paper of Kavan et al.](https://www.cs.utah.edu/~ladislav/kavan10fast/kavan10fast.html)
into hdf5 format to be viewed with view_animation.py.
This is mostly to check if our reimplementation yields the same results as Ladislav's method.
Example:
$ python convert_sma_data.py <path-to-data-from-website>/horse.skin.txt /tmp/horse_sma.h5
$ python view_animation.py /tmp/horse_sma.h5

## view_animation.py

Can view reconstructed animations as well as their estimated bone positions.
