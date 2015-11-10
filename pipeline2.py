from pyIMS.ion_datacube import ion_datacube
from pyMS.pyisocalc import pyisocalc

from itertools import product
import os
import sys
import cPickle
import numpy as np
import scipy as sp
import scipy.sparse as ssp
import scikits.sparse.cholmod as ssc
import logging
import h5py
from numba import njit
from math import sqrt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
def prepare(mzs_list, intensity_list):
    mzs_list = np.asarray(mzs_list).astype(np.float64)
    intensity_list = np.asarray(intensity_list).astype(np.float32)
    intensity_list[intensity_list < 0] = 0
    return mzs_list, intensity_list

class Spectrum(object):
    def __init__(self, i, mzs, intensities, coords):
        self.index = int(i)
        self.mzs = np.asarray(mzs)
        self.intensities = np.asarray(intensities)
        self.coords = np.asarray(coords)

from pyimzml import ImzMLParser
def readImzML(filename):
    f_in = ImzMLParser.ImzMLParser(filename)       
    for i, coords in enumerate(f_in.coordinates):
        mzs, ints = prepare(*f_in.getspectrum(i))
        if len(coords) == 2:
            coords = (coords[0], coords[1], 0)
        yield Spectrum(i, mzs, ints, map(lambda x: x-1, coords))

def readHDF5(filename):
    hdf = h5py.File(filename, 'r')
    for i in hdf['/spectral_data'].keys():
        tmp_str = "/spectral_data/" + i
        mzs = hdf[tmp_str + '/centroid_mzs/']
        ints = hdf[tmp_str + '/centroid_intensities/']
        coords = hdf[tmp_str + '/coordinates/']
        mzs, ints = prepare(mzs, ints)
        yield Spectrum(i, mzs, ints, map(lambda x: x-1, coords))

class Solver(object):
    def __init__(self, nrows, ncols, pixel_indices, spectra, isotope_patterns, lambda_=1e-10, theta=1e-3, rho=None):
        """ spectra: list of Spectrum objects
            isotope_patterns: list of tuples (mzs, intensities) """
        self.nrows = nrows
        self.ncols = ncols
        self.pixel_indices = pixel_indices
        self.spectra = spectra
        print lambda_, theta, rho

        all_mzs = np.concatenate([pattern[0] for pattern in isotope_patterns])
        # append 'infinity' so that searchsorted always returns indices less than the length
        self.all_mz_int_indices = np.concatenate((np.unique(self.bin_numbers(all_mzs)), [np.iinfo(np.int32).max]))

        logging.info("computing Y matrix")
        self.Y = np.asarray(self.sparse_matrix_from_spectra([(s.mzs, s.intensities) for s in self.spectra]).todense(order='F'))

        logging.info("computing D matrix")
        D = self.sparse_matrix_from_spectra(isotope_patterns, assume_presence=True, normalize=True)
        self.D = ssp.hstack((D, 1.0 / np.sqrt(D.shape[0]) * np.ones((D.shape[0], 1)))).tocsr()
        self.D_T = self.D.T.tocsr()

        self.n_masses, self.n_molecules = self.D.shape
        self.n_spectra = self.Y.shape[1]

        logging.info("computing G matrix")
        neighbor_mask = [(x, y, 1.0 / (abs(x) + abs(y))) for x, y in \
                         #[(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
                         [(-1, 0), (1, 0), (0, -1), (0, 1)]
                         ]
        self.G = self.compute_neighbor_matrix(neighbor_mask)

        self.z = self.Y + 1.0
        self.w = np.zeros((self.n_molecules, self.n_spectra))
        #self.w[-1,:] = 1e-4
        self.u_w = np.zeros(self.w.shape)
        self.u_z = np.zeros(self.z.shape)
        self.z_upd_buf = np.zeros(self.z.shape)

        self.w_hat = None

        self.lambda_ = float(lambda_)
        self.theta = float(theta)

        # step size for computing spatial penalty gradient
        if rho is not None:
            self.rho = rho
            self.gamma = 1.0 / rho
        else:
            self.gamma = 1.0 / (2.0 * self.theta * self.G.shape[1])
            self.rho = 1.0 / self.gamma

        logging.info("computing decomposition of (I + D^T . D)")
        self.projector = ssc.cholesky_AAt(self.D_T, 1)

        self.step_size_factors = []

        logging.info("initialized solver")

    def bin_numbers(self, mzs):
        return (mzs * 200).astype(np.int32)

    def sparse_matrix_from_spectra(self, data, assume_presence=False, normalize=False):
        intensity_list = []
        row_list = []
        len_list = []
        for j, (mzs, intensities) in enumerate(data):
            int_mzs = self.bin_numbers(mzs)
            idx = self.all_mz_int_indices.searchsorted(int_mzs)
            if not assume_presence:
                known = np.where(self.all_mz_int_indices[idx] == int_mzs)[0]
                intensities = intensities[known]
                idx = idx[known]
                length = len(known)
            else:
                length = len(mzs)
            if normalize:
                intensities /= intensities.sum()
            intensity_list.append(intensities)
            row_list.append(idx)
            len_list.append(length)

        intensities = np.concatenate(intensity_list)
        rows = np.concatenate(row_list)
        columns = np.repeat(np.arange(len(data), dtype=np.int32), len_list)
        return ssp.coo_matrix((intensities, (rows, columns)),
                              shape=(len(self.all_mz_int_indices), len(data)),
                              dtype=float)

    def compute_neighbor_matrix(self, neighbor_mask):
        neighbors_map = {}
        indices = -1 * np.ones((self.nrows, self.ncols), dtype=int)
        for s in self.spectra:
            x, y = s.coords[:2]
            indices[x, y] = s.index
        for x in xrange(self.nrows):
            for y in xrange(self.ncols):
                if indices[x, y] == -1:
                    continue
                neighbors_map[indices[x, y]] = []
                for dx, dy, w in neighbor_mask:
                    if 0 <= x + dx < self.nrows and 0 <= y + dy < self.ncols:
                        idx = indices[x + dx, y + dy]
                        if idx == -1:
                            continue
                        neighbors_map[indices[x, y]].append((idx, w))

        n_pairs = sum(len(x) for x in neighbors_map.values()) / 2
        rows = np.concatenate([[i, j] for i in neighbors_map for j, w in neighbors_map[i] if i < j])
        columns = np.repeat(np.arange(n_pairs), 2)
        weights = np.concatenate([[w, -w] for i in neighbors_map for j, w in neighbors_map[i] if i < j])
        return ssp.coo_matrix((weights, (rows, columns)),
                              shape=(self.n_spectra, n_pairs),
                              dtype=np.float)

    def project(self, w, z):
        """ projects arbitrary (w, z) pair onto the subspace Dw=z """
        v = w + self.D_T.dot(z)
        w = self.projector(v)
        z = self.D.dot(w)
        return w, z

    def w_w0_prox(self, w_w0_prev):
        w_prev = w_w0_prev[:-1,:]
        w0_prev = w_w0_prev[-1,:]
        return np.vstack((np.maximum(w_prev - self.lambda_ / self.rho, 0.0),
                          np.maximum(w0_prev, 0.0)))

    def z_prox(self, z_prev):
        @njit
        def z_prox_fast_numba(z, Y, rho, out):
            M, N = z.shape
            for i in xrange(M):
                for j in xrange(N):
                    tmp = z[i, j] - 1.0 / rho
                    out[i, j] = 0.5 * (sqrt(tmp * tmp + 4.0 * Y[i, j] / rho) + tmp)

        z_prox_fast_numba(z_prev, self.Y, self.rho, self.z_upd_buf)
        return self.z_upd_buf

    def spatial_penalty_gradient(self, w):
        return np.vstack((2.0 * self.theta * self.G.dot(self.G.T).dot(w[:-1,:].T).T,
                          np.zeros((1, w.shape[1]))))

    def LL(self, w):
        return (self.Y * np.log(self.D.dot(w) + 1e-32)).sum() - self.D.dot(w).sum() - self.lambda_ * w[:-1,:].sum() - self.theta * np.linalg.norm(self.G.T.dot(w[:-1,:].T))**2

    def run_single_iteration(self, safe_step_size=False):
        w1 = self.w_w0_prox(self.u_w)
        z1 = self.z_prox(self.u_z)
        self.w_hat = w1[:]
        if safe_step_size:
            self.w, self.z = self.compute_projection(self.gamma, w1, z1, self.u_w, self.u_z)
        else:
            self.w, self.z = self.project(2 * w1 - self.u_w - self.gamma * self.spatial_penalty_gradient(w1), 2 * z1 - self.u_z)
        self.u_w += 1.0 * (self.w - w1)
        self.u_z += 1.0 * (self.z - z1)

    def get_image(self, w, molecule_index):
        abundancies = w.reshape((self.n_molecules, self.n_spectra), order='F')[molecule_index]
        img = np.zeros(self.nrows * self.ncols)
        img[self.pixel_indices] = abundancies
        return img.reshape((self.nrows, self.ncols))

    def compute_projection(self, gamma, W0, Z0, U_W, U_Z):
        def fprod(A, B):
            return np.dot(A.ravel(), B.ravel())

        def fnorm2(W, Z):
            return np.linalg.norm(W) ** 2 + np.linalg.norm(Z) ** 2

        W1, Z1 = self.project(W0, Z0)
        W1 = W1 - W0
        Z1 -= Z0
        W2, Z2 = self.project(W0 - U_W - 2.0 * gamma * self.theta * self.G.dot(self.G.T).dot(W0.T).T, Z0 - U_Z)
        a = np.linalg.norm(self.G.T.dot(W2.T)) ** 2
        b = 2.0 * fprod(self.G.T.dot(W1.T), self.G.T.dot(W2.T)) - 1.0 / (2.0 * gamma * self.theta) * fnorm2(W2, Z2)
        c = np.linalg.norm(self.G.T.dot(W1.T)) ** 2 - 1.0 / (gamma * self.theta) * (fprod(W1, W2) + fprod(Z1, Z2))
        d = -1.0 / (2.0 * gamma * self.theta) * fnorm2(W1, Z1)
        all_roots = np.roots([a, b, c, d])
        positive_roots = sorted([r.real for r in all_roots if r.real > 0])
        step_size_factor = positive_roots[0]
        self.step_size_factors.append(step_size_factor)
        if step_size_factor > 1.0:
            step_size_factor = 1.0
        return W1 + W0 + step_size_factor * W2, Z1 + Z0 + step_size_factor * Z2

class ProbPipeline(object):
    def __init__(self, config):
        self.config = config
        self.data_file = config['file_inputs']['data_file']

    def initialize(self):
        logging.info("==== computing isotope patterns")
        self.load_queries()
        logging.info("==== loading data")
        self.load_data()

    # creates the output directory if it doesn't exist
    def output_directory(self):
        output_dir = self.config['file_inputs']['results_folder']
        if os.path.isdir(output_dir) == False:
            os.mkdir(output_dir)
        return output_dir

    # config.file_inputs.database_file must contain one formula per line
    def load_queries(self):
        config = self.config
        db_filename = config['file_inputs']['database_file']
        db_dump_folder = config['file_inputs']['database_load_folder']  
        isocalc_sig = config['isotope_generation']['isocalc_sig']  
        isocalc_resolution = config['isotope_generation']['isocalc_resolution']  
        if len(config['isotope_generation']['charge']) > 1:
            print 'Warning: only first charge state currently accepted'
        charge = int('{}{}'.format(config['isotope_generation']['charge'][0]['polarity'], config['isotope_generation']['charge'][0]['n_charges'])) #currently only supports first charge!!
        self.adducts=[a['adduct'] for a in config['isotope_generation']['adducts']]
      
        # Read in molecules
        self.sum_formulae = [l.strip() for l in open(db_filename).readlines()]
        # Check if already generated and load if possible, otherwise calculate fresh   
        db_name =  os.path.splitext(os.path.basename(db_filename))[0] 
        self.mz_list={}
        for adduct in self.adducts:
            for sum_formula in self.sum_formulae:
                isotope_ms = pyisocalc.isodist(sum_formula + adduct,
                                               plot=False,
                                               sigma=isocalc_sig,
                                               charges=charge,
                                               resolution=isocalc_resolution,
                                               do_centroid=True)
                if not sum_formula in self.mz_list:
                     self.mz_list[sum_formula] = {}

                mzs, ints = map(np.array, isotope_ms.get_spectrum(source='centroids'))
                order = ints.argsort()[::-1]
                self.mz_list[sum_formula][adduct] = (mzs[order], ints[order])

    def _calculate_dimensions(self):
        cube = ion_datacube()
        cube.add_coords(self.coords)
        self.nrows = int(cube.nRows)
        self.ncols = int(cube.nColumns)
        self.pixel_indices = cube.pixel_indices

    def load_data(self):
        if self.data_file.endswith(".imzML"):
            spectra = readImzML(self.data_file)
        elif self.data_file.endswith(".hdf5"):
            spectra = readHDF5(self.data_file)
        else:
            raise "the input format is unsupported"

        self.spectra = list(spectra)

        self.coords = np.zeros((len(self.spectra), 3))
        for i, sp in enumerate(self.spectra):
            self.coords[i, :] = sp.coords
        self._calculate_dimensions()

    def get_solver(self, **kwargs):
        isotope_patterns = [self.mz_list[f][a] for f in self.sum_formulae for a in self.adducts]
        return Solver(self.nrows, self.ncols, self.pixel_indices, self.spectra, isotope_patterns, **kwargs)
