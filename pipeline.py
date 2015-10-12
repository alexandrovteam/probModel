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

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
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
            return (mzs * 1000).astype(np.int32)

        isotope_patterns = [self.mz_list[f][a] for f in self.sum_formulae for a in self.adducts]
        all_mzs = np.concatenate([pattern[0] for pattern in isotope_patterns])
        # append 'infinity' so that searchsorted always returns indices less than the length
        all_mz_int_indices = np.concatenate((np.unique(bin_numbers(all_mzs)), [np.iinfo(np.int32).max]))

        def sparse_matrix_from_spectra(data, assume_presence=False):
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
        Y = sparse_matrix_from_spectra([(s.mzs, s.intensities) for s in self.spectra])
        #print Y.nnz, Y.shape
        logging.info("computing D matrix")
        D = sparse_matrix_from_spectra(isotope_patterns, assume_presence=True)
        #print D.nnz, D.shape

        n_masses, n_molecules = D.shape
        n_spectra = Y.shape[1]

        #np.set_printoptions(threshold='nan', linewidth=300)
        #print (D.todense() > 0).astype(int)

        neighbors_map = {}
        indices = -1 * np.ones((self.nrows, self.ncols), dtype=int)
        for s in self.spectra:
            x, y = s.coords[:2]
            indices[x, y] = s.index
        for x in xrange(self.nrows):
            for y in xrange(self.ncols):
                neighbors_map[indices[x, y]] = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    if 0 <= x + dx < self.nrows and 0 <= y + dy < self.ncols:
                        idx = indices[x + dx, y + dy]
                        if idx == -1:
                            continue
                        neighbors_map[indices[x, y]].append(idx)

        n_pairs = sum(len(x) for x in neighbors_map.values()) / 2

        def w_w0_update_matrix():
            xs = []
            ys = []
            data = []

            # upper part (corresponds to DW + W0)
            for i in xrange(n_spectra):
                y_offset = n_molecules * i
                x_offset = n_masses * i

                ys.append(D.col + y_offset)
                xs.append(D.row + x_offset)
                data.append(D.data)

            ys.append(np.repeat(np.arange(n_spectra) + n_molecules * n_spectra, n_masses))
            xs.append(np.arange(n_masses * n_spectra))
            data.append(np.ones(n_masses * n_spectra))

            # middle part (corresponds to W)
            x_offset = n_masses * n_spectra
            
            ys.append(np.arange(n_molecules * n_spectra))
            xs.append(np.arange(n_molecules * n_spectra) + x_offset)
            data.append(np.ones(n_molecules * n_spectra))

            # lower part (corresponds to the neighbor abundancy differences)
            x_offset = (n_masses + n_molecules) * n_spectra

            for i in neighbors_map:
                for j in neighbors_map[i]:
                    if i > j: continue
                    ys.append(np.arange(n_molecules) + n_molecules * i)
                    xs.append(np.arange(n_molecules) + x_offset)
                    data.append(np.ones(n_molecules))

                    ys.append(np.arange(n_molecules) + n_molecules * j)
                    xs.append(np.arange(n_molecules) + x_offset)
                    data.append(-1 * np.ones(n_molecules))
                    x_offset += n_molecules
            
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            data = np.concatenate(data)
            
            result = ssp.coo_matrix((data, (xs, ys)), dtype=float)

            assert result.nnz == (D.nnz + n_masses + n_molecules) * n_spectra + n_molecules * n_pairs * 2
            assert result.shape[0] == (n_molecules + n_masses) * n_spectra + n_pairs * n_molecules
            assert result.shape[1] == n_spectra * (n_molecules + 1)

            return result.tocsc()

        A = w_w0_update_matrix()
        print A.shape, A.nnz

        Y = Y.todense().A1
        
        z0 = Y + 1
        u0 = 1e-4 * np.ones(n_masses * n_spectra)

        z1 = 1e-4 * np.ones(n_molecules * n_spectra)
        u1 = 1e-4 * np.ones(n_molecules * n_spectra)

        z2 = 1e-4 * np.ones(n_pairs * n_molecules)
        u2 = 1e-4 * np.ones(n_pairs * n_molecules)

        from sklearn.linear_model import Lasso, ElasticNet

        lambda_ = 1e-4
        theta = 1e-4
        rho = 1e-2

        print lambda_/rho/A.shape[0]

        w_w0_lasso = Lasso(alpha=lambda_/rho/A.shape[0], fit_intercept=False, positive=True)
        z1_lasso = Lasso(alpha=lambda_/rho/z1.shape[0], fit_intercept=False, warm_start=True, positive=True)
        z2_ridge = ElasticNet(alpha=theta/rho/z2.shape[0], l1_ratio=0, warm_start=True, positive=True, fit_intercept=False)

        def w_w0_update():
            rhs = np.concatenate((z0 + 1.0/rho * u0, z1 + 1.0/rho * u1, z2 + 1.0/rho * u2))
            w_w0_lasso.fit(A, rhs)
            w = w_w0_lasso.coef_[:n_molecules*n_spectra]
            w0 = w_w0_lasso.coef_[n_molecules*n_spectra:]
            return w, w0

        from scipy.optimize import fmin_l_bfgs_b
        def z0_update(Dw_w0, u0):
            def f(x):
                return np.dot(u0 + 1, x) - np.dot(Y, np.log(x)) + rho/2 * ((x - Dw_w0)**2).sum()

            def g(x):
                return u0 + 1 - Y / x + rho * (x - Dw_w0)

            x, value, d = fmin_l_bfgs_b(f, z0, g)
            return x

        def z1_update(w, u1):
            z1_lasso.fit(ssp.eye(z1.shape[0]), w - 1.0 / rho * u1)
            return z1_lasso.coef_

        def z2_update(diffs, u2):
            z2_ridge.fit(ssp.eye(z2.shape[0]), diffs*diffs - 1.0 / rho * u2)
            return z2_ridge.coef_

        def LL(w, Dw_w0, diffs):
            return np.dot(Y, Dw_w0) - Dw_w0.sum() - lambda_ * w.sum() - theta * np.linalg.norm(diffs)**2

        max_iter = 100
        for i in range(max_iter):
            logging.info("w,w0 update")
            w, w0 = w_w0_update()
            rhs = w_w0_lasso.predict(A)
            Dw_w0_estimate = rhs[:n_masses*n_spectra]
            w_estimate = rhs[n_masses*n_spectra:(n_masses+n_molecules)*n_spectra]
            diff_estimates = rhs[(n_masses+n_molecules)*n_spectra:]
            print LL(w_estimate, Dw_w0_estimate, diff_estimates)
            logging.info("z0 update")
            z0 = z0_update(Dw_w0_estimate, u0)
            print LL(w_estimate, z0, diff_estimates)
            logging.info("z1 update")
            z1 = z1_update(w, u1)
            print LL(z1, z0, diff_estimates)
            logging.info("z2 update")
            z2 = z2_update(diff_estimates, u2)
            u0 += rho * (z0 - Dw_w0_estimate)
            u1 += rho * (z1 - w_estimate)
            u2 += rho * (z2 - diff_estimates**2)
            print LL(z1, z0, z2)
            print w.reshape((n_molecules, n_spectra)).sum(axis=1)
            print w0.sum()


if __name__ == '__main__':
    import json
    import sys
    config = json.loads(open(sys.argv[1]).read())
    pipeline = ProbPipeline(config)
    pipeline.run()
