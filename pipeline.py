from pyIMS.ion_datacube import ion_datacube
from pyMS.pyisocalc import pyisocalc

from itertools import product
import os
import sys
import cPickle
import numpy as np
import scipy as sp
import scipy.sparse as ssp
import logging
import h5py

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

class ProbPipeline(object):
    def __init__(self, config):
        self.config = config
        self.data_file = config['file_inputs']['data_file']

    def run(self):
        logging.info("==== computing isotope patterns")
        self.load_queries()
        logging.info("==== loading data")
        self.load_data()
        self.compute_abundancies()

    # template method
    def compute_scores(self):
        ### Runs the main pipeline
        # Get sum formula and predicted m/z peaks for molecules in database
        # Parse dataset
        raise NotImplementedError

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

    def compute_abundancies(self):
        def bin_numbers(mzs):
            return (mzs * 200).astype(np.int32)

        isotope_patterns = [self.mz_list[f][a] for f in self.sum_formulae for a in self.adducts]
        all_mzs = np.concatenate([pattern[0] for pattern in isotope_patterns])
        # append 'infinity' so that searchsorted always returns indices less than the length
        all_mz_int_indices = np.concatenate((np.unique(bin_numbers(all_mzs)), [np.iinfo(np.int32).max]))

        def sparse_matrix_from_spectra(data, assume_presence=False, normalize=False):
            intensity_list = []
            row_list = []
            len_list = []
            for j, (mzs, intensities) in enumerate(data):
                int_mzs = bin_numbers(mzs)
                idx = all_mz_int_indices.searchsorted(int_mzs)
                if not assume_presence:
                    known = np.where(all_mz_int_indices[idx] == int_mzs)[0]
                    intensities = intensities[known]
                    idx = idx[known]
                    length = len(known)
                else:
                    length = len(mzs)
                if normalize:
                    intensities /= np.linalg.norm(intensities)
                intensity_list.append(intensities)
                row_list.append(idx)
                len_list.append(length)

            intensities = np.concatenate(intensity_list)
            rows = np.concatenate(row_list)
            columns = np.repeat(np.arange(len(data), dtype=np.int32), len_list)
            result = ssp.coo_matrix((intensities, (rows, columns)),
                                     shape=(len(all_mz_int_indices), len(data)),
                                     dtype=float)
            return result

        #print self.sum_formulae
        logging.info("computing Y matrix")
        Y = np.asarray(sparse_matrix_from_spectra([(s.mzs, s.intensities) for s in self.spectra]).todense(order='F'))
        #print Y.nnz, Y.shape
        logging.info("computing D matrix")
        D = sparse_matrix_from_spectra(isotope_patterns, assume_presence=True, normalize=True)
        D = ssp.hstack((D, 1.0 / np.sqrt(D.shape[0]) * np.ones((D.shape[0], 1)))).tocsr()
        D_T = D.T.tocsr()
        #print D.nnz, D.shape

        n_masses, n_molecules = D.shape
        n_spectra = Y.shape[1]

        np.set_printoptions(threshold='nan', linewidth=300, precision=3, suppress=True)
        #print (D.todense() > 0).astype(int)

        neighbors_map = {}
        indices = -1 * np.ones((self.nrows, self.ncols), dtype=int)
        for s in self.spectra:
            x, y = s.coords[:2]
            indices[x, y] = s.index
        for x in xrange(self.nrows):
            for y in xrange(self.ncols):
                neighbors_map[indices[x, y]] = []
                #for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= x + dx < self.nrows and 0 <= y + dy < self.ncols:
                        idx = indices[x + dx, y + dy]
                        if idx == -1:
                            continue
                        neighbors_map[indices[x, y]].append(idx)

        n_pairs = sum(len(x) for x in neighbors_map.values()) / 2

        z = Y
        w = 1e-4 * np.ones((D.shape[1], Y.shape[1]))
        u_w = np.zeros(w.shape)
        u_z = np.zeros(z.shape)

        lambda_ = 1e-10
        # TODO: test with spatial penalties
        theta = 0 #1e-5
        rho = 10.0

        neighbors_map = {}

        import scipy.sparse.linalg as sspl
        projector = sspl.splu(sp.sparse.eye(D.shape[1]) + D.T.dot(D))

        def project(w, z):
            v = w + D_T.dot(z)
            w = projector.solve(v)
            z = D.dot(w)
            return w, z

        def w_w0_prox(w_w0_prev):
            x = np.maximum(w_w0_prev - lambda_ / rho, 0.0)
            return x
            w_w0 = x[:]
            x_c = w_w0_prev - lambda_ / 2.0 / rho
            for i in neighbors_map:
                for j in neighbors_map[i]:
                    if i > j: continue
                    w_w0[:, i] -= x[:, i] / n_pairs
                    w_w0[:, j] -= x[:, j] / n_pairs
                    x_c_i, x_c_j = x_c[:, i].T, x_c[:, j].T
                    b0 = (x_c_i >= 0) & (x_c_j >= 0) # interior
                    b1 = (x_c_i >= 0) & (x_c_j <  0) # border 1
                    b2 = (x_c_i  < 0) & (x_c_j >= 0) # border 2
                    x_c_diff = x_c_j - x_c_i
                    w_w0[b0, i] += (x_c_i[b0] + 2.0 * theta / (4.0 * theta + rho) * x_c_diff[b0]) / n_pairs
                    w_w0[b0, j] += (x_c_j[b0] - 2.0 * theta / (4.0 * theta + rho) * x_c_diff[b0]) / n_pairs
                    w_w0[b1, i] += x_c_i[b1] / (2.0 * theta / rho + 1.0) / n_pairs
                    w_w0[b2, j] += x_c_j[b2] / (2.0 * theta / rho + 1.0) / n_pairs
            return w_w0

        def z_prox(z_prev):
            tmp = z_prev - 1.0 / rho
            return 0.5 * (np.sqrt(tmp ** 2 + 4.0 * Y / rho) + tmp)

        from numba import njit
        from math import sqrt

        @njit
        def z_prox_fast_numba(z, Y, rho, out):
            M, N = z.shape
            #r = np.zeros((M, N))
            for i in xrange(M):
                for j in xrange(N):
                    tmp = z[i, j] - 1.0 / rho
                    out[i, j] = 0.5 * (sqrt(tmp * tmp + 4.0 * Y[i, j] / rho) + tmp)
            #return r

        def z_prox_fast(z_prev, out):
            z_prox_fast_numba(z_prev, Y, rho, out)
            return out

        def LL(w):
            return (Y * np.log(D.dot(w) + 1e-32)).sum() - D.dot(w).sum() - lambda_ * w.sum()

        max_iter = 2000
        old_w1 = None
        tmp = np.zeros(z.shape)

        for i in range(max_iter):
            w1 = w_w0_prox(u_w)
            z1 = z_prox_fast(u_z, tmp)
            if old_w1 is not None and i % 20 == 0 and i > 0:
                logging.info("%.3f, %.3f, %.3f" % (LL(w1), np.linalg.norm(w1 - old_w1), np.linalg.norm(w1)))
            old_w1 = w1[:]
            w, z = project(2 * w1 - u_w, 2 * z1 - u_z)
            u_w += 1.5 * (w - w1)
            u_z += 1.5 * (z - z1)
            if i % 100 == 0 and i > 0:
                print w_w0_prox(u_w).reshape((n_molecules, self.nrows, self.ncols), order='F').sum(axis=(1,2))

if __name__ == '__main__':
    import json
    import sys
    config = json.loads(open(sys.argv[1]).read())
    pipeline = ProbPipeline(config)
    pipeline.run()
