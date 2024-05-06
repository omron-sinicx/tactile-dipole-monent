'''
Copyright (c) 2015, Harsh Bhatia (bhatia4@llnl.gov)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
from .poisson import PoissonSolver
from .structured import StructuredGrid
from .unstructured import UnstructuredGrid
from .spherical import StructuredSphericalGrid

# ---------------------------------------------------------------------------
class nHHD(object):

    # constructor
    def __init__(self, **kwargs):

        args = list(kwargs.keys())

        if not ('grid' in args) and not ('points' in args) and not ('sphericalgrid' in args):
            raise SyntaxError("nHHD Solver needs either shape of a regular grid or the points in an rectilinear/unstructured grid")
        if (('grid' in args) != ('spacings' in args)) and (('sphericalgrid' in args) != ('spacings' in args)):
            raise SyntaxError("nHHD needs spacings for regular grid")

        if 'grid' in args:
            self.mesh = StructuredGrid(grid=kwargs['grid'], spacings=kwargs['spacings'])
            self.psolver = PoissonSolver(grid=self.mesh.dims, spacings=self.mesh.dx)

        elif 'points' in args:
            if 'simplices' in args:
                self.mesh = UnstructuredGrid(vertices=kwargs['points'], simplices=kwargs['simplices'])
            else:
                self.mesh = UnstructuredGrid(vertices=kwargs['points'])
            self.psolver = PoissonSolver(points=self.mesh.vertices, pvolumes=self.mesh.pvolumes)

        elif 'sphericalgrid' in args:
            self.mesh = StructuredSphericalGrid(sphericalgrid=kwargs['sphericalgrid'],
                                                spacings=kwargs['spacings'],
                                                lat=kwargs['lat'], lon=kwargs['lon'])
            self.psolver = PoissonSolver(sphericalgrid=self.mesh.dims, spacings=self.mesh.dx,
                                         lat=kwargs['lat'], lon=kwargs['lon'])

        self.dim = self.mesh.dim
        if (self.dim != 2) and (self.dim != 3):
            raise ValueError("nHHD Solver works only for 2 and 3 dimensions")

        if not 'sphericalgrid' in args:
            # no need to do this if spherical grid
            # we need to calculate greens's function multiple times
            self.psolver.prepare(True)

    # create the 3-component decomposition
    def decompose(self, vfield, div=None, curlw=None, num_cores=32, tuning_param=0.0625):

        if vfield.shape[-1] != self.dim:
            raise ValueError("nHHD.decompose requires a valid-dimensional vector field")

        # ----------------------------------------------------------------------
        if self.dim == 2:

            # compute div curl
            if np.all(div==None) and np.all(curlw==None):
                # calculate div and curl if they are not known
                (self.div, self.curlw) = self.mesh.divcurl(vfield)
            else:
                # here is a possibility to give div and curl directly
                # highly suggested if your data is not on cartesian grid
                # e.g. with model output, use the model definition of the div and curl
                self.div = div
                self.curlw = curlw

            # compute potentials
            self.nD, self.nRu = self.psolver.solve([self.div, self.curlw],
                                                   num_cores=num_cores,
                                                   tuning_param=tuning_param)

            # compute fields as gradients of potentials
            self.d = self.mesh.gradient(self.nD)
            self.r = self.mesh.rotated_gradient(self.nRu)

        else:
            # compute div curl
            (self.div, self.curlu, self.curlv, self.curlw) = self.mesh.divcurl(vfield)

            # compute potentials
            self.nD, self.nRu, self.nRv, self.nRw = self.psolver.solve([self.div, self.curlu, self.curlv, self.curlw])
            self.nRu *= -1.0
            self.nRv *= -1.0
            self.nRw *= -1.0

            # compute divergent field as gradient of D potential
            self.d = self.mesh.gradient(self.nD)

            # temporary use of r
            self.r = np.stack((self.nRu, self.nRv, self.nRw), axis = -1)

            # compute rotational field as curl of R potential
            (cu, cv, cw) = self.mesh.curl3D(self.r)
            self.r = np.stack((cu, cv, cw), axis = -1)

        self.h = np.add(self.d, self.r)
        np.subtract(vfield, self.h, self.h)

# ---------------------------------------------------------------------------
