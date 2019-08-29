import os
import tables as tb
import numpy  as np

from invisible_cities.io  .mcinfo_io   import read_mcinfo

from invisible_cities.core.exceptions  import SipmEmptyList
from invisible_cities.core.exceptions  import SipmZeroCharge

from invisible_cities.io  .dst_io      import load_dst
from invisible_cities.reco.corrections import Correction



def from_cartesian_to_cyl(pos):
    cyl_pos = np.array([np.sqrt(pos[:,0]**2+pos[:,1]**2), np.arctan2(pos[:,1], pos[:,0]), pos[:,2]]).transpose()
    return cyl_pos


def barycenter_3D(pos, qs):
    """Returns the weighted position of an array
    """
    if not len(pos)   : raise SipmEmptyList
    if np.sum(qs) == 0: raise SipmZeroCharge

    return np.average(pos, weights=qs, axis=0)


def load_rpos(filename, group = "Radius", node = "f100bins"):
    dst = load_dst(filename, group, node)
    return Correction((dst.Sigma_phi.values,), dst.Rpos.values, dst.Uncertainty.values)


def load_zr_corrections(filename, *,
                        group = "Corrections",
                        node  = "ZRcorrections",
                        **kwargs):
    dst  = load_dst(filename, group, node)
    z, r = np.unique(dst.z.values), np.unique(dst.r.values)
    f, u = dst.factor.values, dst.uncertainty.values

    return Correction((z, r),
                      f.reshape(z.size, r.size),
                      u.reshape(z.size, r.size),
                      **kwargs)
