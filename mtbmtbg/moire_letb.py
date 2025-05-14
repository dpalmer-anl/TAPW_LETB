import h5py
import numpy as np
import pandas as pd
import sys
from scipy.spatial.distance import cdist
import bilayer_letb
from bilayer_letb.functions import fang, moon
import matplotlib.pyplot as plt

def ix_to_orientation(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Converts displacement indices to orientations of the 
    nearest neighbor environments using definitions in 
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    theta_12 - Orientation of upper-layer relative to bond length
    theta_21 - Orientation of lower-layer relative to bond length
    """
    import scipy.spatial as spatial
    displacement_vector = di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]
    mat = nnmat(lattice_vectors, atomic_basis)

    # Compute distances and angles
    theta_12 = []
    theta_21 = []
    for disp, i, j, inn, jnn in zip(displacement_vector, ai, aj, mat[ai], mat[aj]):
        sin_jnn = np.cross(jnn[:,:2], disp[:2]) 
        sin_inn = np.cross(inn[:,:2], disp[:2]) 
        cos_jnn = np.dot(jnn[:,:2], disp[:2]) 
        cos_inn = np.dot(inn[:,:2], disp[:2]) 
        theta_jnn = np.arctan2(sin_jnn, cos_jnn)
        theta_inn = np.arctan2(sin_inn, cos_inn)

        theta_12.append(np.pi - theta_jnn[0])
        theta_21.append(theta_inn[0])
    return theta_12, theta_21

def interlayer_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    
    output = {
        'dxy': [], # Distance in Bohr, xy plane
        'dz': [],  # Distance in Bohr, z
        'd': [],   # Distance in Bohr 
        'theta_12': [], # Orientation of upper layer NN environment
        'theta_21': [], # Orientation of lower layer NN environment
    }

    # 1-body terms
    dist_xy, dist_z = ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj)
    dist = np.sqrt(dist_z ** 2 + dist_xy ** 2)
    output['dxy'] = list(dist_xy)
    output['dz'] = list(dist_z)
    output['d'] = list(dist)

    # Many-body terms
    theta_12, theta_21 = ix_to_orientation(lattice_vectors, atomic_basis, di, dj, ai, aj)
    output['theta_12'] += list(theta_12)
    output['theta_21'] += list(theta_21)
   
    # Return pandas DataFrame
    df = pd.DataFrame(output)
    return df

def nnmat(lattice_vectors, atomic_basis):
    """
    Build matrix which tells you relative coordinates
    of nearest neighbors to an atom i in the supercell

    Returns: nnmat [natom x 3 x 3]
    """
    import scipy.spatial as spatial
    nnmat = np.zeros((len(atomic_basis), 3, 3))

    # Extend atom list
    atoms = []
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            displaced_atoms = atomic_basis + lattice_vectors[np.newaxis, 0] * i + lattice_vectors[np.newaxis, 1] * j
            atoms += [list(x) for x in displaced_atoms]
    atoms = np.array(atoms)
    atomic_basis = np.array(atomic_basis)

    # Loop
    for i in range(len(atomic_basis)):
        displacements = atoms - atomic_basis[i]
        distances = np.linalg.norm(displacements,axis=1)
        ind = np.argsort(distances)
        nnmat[i] = displacements[ind[1:4]]

    return nnmat

def ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """ 
    Converts displacement indices to physical distances
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    dxy - Distance in Bohr, projected in the x/y plane
    dz  - Distance in Bohr, projected onto the z axis
    """
    displacement_vector = di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]

    displacement_vector_xy = displacement_vector[:, :2] 
    displacement_vector_z =  displacement_vector[:, -1] 

    dxy = np.linalg.norm(displacement_vector_xy, axis = 1)
    dz = np.abs(displacement_vector_z)
    return dxy, dz

def partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Given displacement indices and geometry,
    get indices for partitioning the data
    """
    # First find the smallest distance in the lattice -> reference for NN 
    distances = ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj)
    distances = np.sqrt(distances[0]**2 + distances[1]**2)
    min_distance = min(distances)

    # NN should be within 5% of min_distance
    t01_ix = (distances >= 0.95 * min_distance) & (distances <= 1.05 * min_distance)

    # NNN should be withing 5% of sqrt(3)x of min_distance
    t02_ix = (distances >= 0.95 * np.sqrt(3) * min_distance) & (distances <= 1.05 * np.sqrt(3) * min_distance)

    # NNNN should be within 5% of 2x of min_distance
    t03_ix = (distances >= 0.95 * 2 * min_distance) & (distances <= 1.05 * 2 * min_distance)
   
    # Anything else, we zero out
    t00 = (distances < 0.95 * min_distance) | (distances > 1.05 * 2 * min_distance)

    return t01_ix, t02_ix, t03_ix, t00

def triangle_height(a, base):
    """
    Give area of a triangle given two displacement vectors for 2 sides
    """
    area = np.linalg.det(
            np.array([a, base, [1, 1, 1]])
    )
    area = np.abs(area)/2
    height = 2 * area / np.linalg.norm(base)
    return height

def t01_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    # Compute NN distances
    r = di[:, np.newaxis] * lattice_vectors[0] + dj[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai] # Relative coordinates
    #r[:, -1] = 0 # Project into xy-plane
    a = np.linalg.norm(r, axis = 1)
    return pd.DataFrame({'a': a})

def t02_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    # Compute NNN distances
    r = di[:, np.newaxis] * lattice_vectors[0] + dj[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai] # Relative coordinates
    #r[:, -1] = 0 # Project into xy-plane
    b = np.linalg.norm(r, axis = 1)

    # Compute h
    h1 = []
    h2 = []
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in range(len(r)):
        nn = mat[aj[i]] + r[i]
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        h1.append(triangle_height(nn[ind[0]], r[i]))
        h2.append(triangle_height(nn[ind[1]], r[i]))
    return pd.DataFrame({'h1': h1, 'h2': h2, 'b': b})
    
def t03_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Compute t03 descriptors
    """
    # Compute NNNN distances
    r = di[:, np.newaxis] * lattice_vectors[0] + dj[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai] # Relative coordinates
    c = np.linalg.norm(r, axis = 1)
    #r[:, -1] = 0 # Project into xy-plane

    # All other hexagon descriptors
    l = []
    h = []
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in range(len(r)):
        nn = mat[aj[i]] + r[i]
        nn[:, -1] = 0 # Project into xy-plane
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        b = nndist[ind[0]]
        d = nndist[ind[1]]
        h3 = triangle_height(nn[ind[0]], r[i])
        h4 = triangle_height(nn[ind[1]], r[i])

        nn = r[i] - mat[ai[i]]
        nn[:, -1] = 0 # Project into xy-plane
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        a = nndist[ind[0]]
        e = nndist[ind[1]]
        h1 = triangle_height(nn[ind[0]], r[i])
        h2 = triangle_height(nn[ind[1]], r[i])

        l.append((a + b + d + e)/4)
        h.append((h1 + h2 + h3 + h4)/4)
    return pd.DataFrame({'c': c, 'h': h, 'l': l})

def intralayer_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """ 
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    # Partition 
    partition = partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj)
    
    # Compute descriptors
    t01 = t01_descriptors(lattice_vectors, atomic_basis, di[partition[0]], dj[partition[0]], ai[partition[0]], aj[partition[0]])
    t02 = t02_descriptors(lattice_vectors, atomic_basis, di[partition[1]], dj[partition[1]], ai[partition[1]], aj[partition[1]])
    t03 = t03_descriptors(lattice_vectors, atomic_basis, di[partition[2]], dj[partition[2]], ai[partition[2]], aj[partition[2]])
    return t01, t02, t03

def load_intralayer_fit():
    # Load in fits, average over k-folds
    fit = {}
    f = "/".join(bilayer_letb.__file__.split("/")[:-1])+"/parameters/fit_intralayer.hdf5"
    with h5py.File(f,'r') as hdf:
        fit['t01'] = np.array(list(hdf['t01']['parameters_test'])).mean(axis = 0)
        fit['t02'] = np.array(list(hdf['t02']['parameters_test'])).mean(axis = 0)
        fit['t03'] = np.array(list(hdf['t03']['parameters_test'])).mean(axis = 0)
    return fit

def load_interlayer_fit():
    # Load in fits, average over k-folds
    fit = {}
    f = "/".join(bilayer_letb.__file__.split("/")[:-1])+"/parameters/fit_interlayer.hdf5"
    with h5py.File(f,'r') as hdf:
        fit['fang'] = np.array(list(hdf['fang']['parameters_test'])).mean(axis = 0)
    return fit

def intralayer(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    Our model for single layer intralayer
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for intralayer in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj in eV
    """
    # Extend lattice_vectors to (3 x 3) for our descriptors, the third lattice vector is arbitrary
    latt_vecs = np.append(lattice_vectors, [[0, 0, 0]], axis = 0)
    atomic_basis = np.array(atomic_basis)
    i = np.array(i)
    j = np.array(j)
    di = np.array(di)
    dj = np.array(dj)

    # Get the descriptors for the fit models
    partition   = partition_tb(lattice_vectors, atomic_basis, di, dj, i, j)
    descriptors = intralayer_descriptors(lattice_vectors, atomic_basis, di, dj, i, j)

    # Get the fit model parameters
    fit = load_intralayer_fit()

    # Predict hoppings
    t01 = np.dot(descriptors[0], fit['t01'][1:]) + fit['t01'][0]
    t02 = np.dot(descriptors[1], fit['t02'][1:]) + fit['t02'][0]
    t03 = np.dot(descriptors[2], fit['t03'][1:]) + fit['t03'][0]

    # Reorganize
    hoppings = np.zeros(len(i))
    hoppings[partition[0]] = t01
    hoppings[partition[1]] = t02
    hoppings[partition[2]] = t03
    hoppings[partition[3]] = 0

    return hoppings

def letb(atomic_basis,lattice_vectors, npair_dict,layer_types=None):
    """
    Our model for bilayer intralayer
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for intralayer in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    # Extend lattice_vectors to (3 x 3) for our descriptors, the third lattice vector is arbitrary
    ang_per_bohr = 0.529
    lattice_vectors /= ang_per_bohr
    atomic_basis = np.array(atomic_basis) / ang_per_bohr

    natom = len(atomic_basis)

    i = np.array(npair_dict["r"])
    j = np.array(npair_dict["c"])
    di = np.array(npair_dict["di"])
    dj = np.array(npair_dict["dj"])

    # Get the bi-layer descriptors 
    descriptors = interlayer_descriptors(lattice_vectors, atomic_basis, di, dj, i, j)
    
    # Partition the intra- and inter-layer hoppings indices 
    if type(layer_types)==np.ndarray or type(layer_types)==list:
        #fix this
        npairs = np.shape(di)[0]
        interlayer = np.full((npairs), False)
        for n in range(npairs): 
            if layer_types[i[n]]!=layer_types[j[n]]:
                interlayer[n]=True 

    else:
        interlayer = np.array(descriptors['dz'] > 1) # Allows for buckling, doesn't work for large corrugation
    # Compute the inter-layer hoppings
    fit = load_interlayer_fit()
    X = descriptors[['dxy','theta_12','theta_21']].values[interlayer]
    interlayer_hoppings = fang(X.T, *fit['fang'])

    # Compute the intra-layer hoppings
    intralayer_hoppings = intralayer(lattice_vectors, atomic_basis, 
                                     i[~interlayer], j[~interlayer], 
                                     di[~interlayer], dj[~interlayer])

    # Reorganize
    hoppings = np.zeros(len(i))
    hoppings[interlayer] = interlayer_hoppings
    hoppings[~interlayer] = intralayer_hoppings

    return hoppings