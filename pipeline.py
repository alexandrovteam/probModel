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
                intensity_list.append(intensities)#/np.linalg.norm(intensities))
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

        nz = np.where(Y.sum(axis=0)>0)[1]#.A1
        xs= self.coords[nz,0]
        ys= self.coords[nz,1]
        # FIXME: there must be a simpler way!
        Y = Y.todense().A1.reshape((n_masses, n_spectra)).ravel(order='F')
        print "Y sum:", Y.sum()
        
        z0 = Y+1
        u0 = np.zeros(n_masses * n_spectra)

        z1 = np.zeros(n_molecules * n_spectra)
        u1 = np.zeros(n_molecules * n_spectra)

        z2 = np.zeros(n_pairs * n_molecules)
        u2 = np.zeros(n_pairs * n_molecules)

        from sklearn.linear_model import Lasso, ElasticNet, LinearRegression

        lambda_ = 1e-4
        theta = 1e-20
        rho = 1e-6

        print lambda_/rho/A.shape[0]

        w_w0_lasso = Lasso(alpha=lambda_/rho/A.shape[0], warm_start=True, fit_intercept=False, positive=True)
        z1_lasso = Lasso(alpha=lambda_/rho/z1.shape[0], fit_intercept=False, warm_start=True, positive=False)
        z2_ridge = ElasticNet(alpha=2*theta/rho/z2.shape[0], l1_ratio=0, warm_start=True, positive=False, fit_intercept=False)

        def w_w0_update():
            rhs = np.concatenate((z0 + 1.0/rho * u0, z1 + 1.0/rho * u1, z2 + 1.0/rho * u2))
            w_w0_lasso.fit(A, rhs)
            w = w_w0_lasso.coef_[:n_molecules*n_spectra]
            w0 = w_w0_lasso.coef_[n_molecules*n_spectra:]
            return w, w0

        from scipy.optimize import fmin_l_bfgs_b
        import nlopt

        from numba import njit

        def z0_update(Dw_w0, u0):
            eps = 1e-10
            def f(x):
                result = x.sum() + rho/2 * ((x - Dw_w0 + 1/rho * u0)**2).sum()
                log_x = np.log(x)
                log_x[x<eps] = np.log(eps) - 1.5 + 2 * x[x<eps] / eps - x[x<eps]**2 / (2*eps**2)
                result -= np.dot(Y, log_x)
                return result

            def g(x):
                result = -Y / x
                result[x<eps] = -Y[x<eps] / (2.0 / eps - x[x<eps]/eps**2)
                result += 1 + rho * (x - Dw_w0 + 1/rho * u0)
                return result

            @njit
            def fast_f(x, Y, eps, rho, u0, Dw_w0):
                result = 0
                log_eps = np.log(eps)
                for i in xrange(len(x)):
                    result += x[i] + rho/2 * (x[i] - Dw_w0[i] + 1/rho * u0[i]) ** 2
                    if x[i] < eps:
                        result -= Y[i] * (log_eps - 1.5 + 2 * x[i] / eps - x[i]**2 / (2*eps**2))
                    else:
                        result -= Y[i] * np.log(x[i])
                return result

            @njit
            def fast_g(x, Y, eps, rho, u0, Dw_w0): 
                result = np.zeros(x.shape[0])
                for i in xrange(len(result)):
                    if x[i] < eps:
                        result[i] = -Y[i] / (2.0 / eps - x[i] / eps**2)
                    else:
                        result[i] = -Y[i] / x[i]
                    result[i] += 1 + rho * (x[i] - Dw_w0[i] + 1/rho * u0[i])
                return result

            def fg(x, grad):
                grad[:] = fast_g(x, Y, eps, rho, u0, Dw_w0)
                return fast_f(x, Y, eps, rho, u0, Dw_w0)

            #x, value, d = fmin_l_bfgs_b(f, z0, g, iprint=0)
            #return x

            z0_opt = nlopt.opt(nlopt.LD_LBFGS, z0.shape[0])
            z0_opt.set_lower_bounds(0)
            z0_opt.set_min_objective(fg)
            z0_opt.set_ftol_abs(2e-9)
            z0_opt.set_maxeval(10)
            result = z0_opt.optimize(z0)
            return result

        def z1_update(w, u1):
            z1_lasso.fit(ssp.eye(z1.shape[0]), w - 1.0 / rho * u1)
            return z1_lasso.coef_

        def z2_update(diffs, u2):
            z2_ridge.fit(ssp.eye(z2.shape[0]), diffs - 1.0 / rho * u2)
            return z2_ridge.coef_

        def logdot(x, y):
            #if np.any((x>0)&(y==0)):
            #    return -np.inf
            return np.dot(x, np.log(y+1e-32))

        # log-likelihood for the original problem (w, w0 variables) 
        def LL(w, Dw_w0=None, diffs=None, w0=None):
            if Dw_w0 is None or diffs is None:
                assert w0 is not None
                rhs = A.dot(np.hstack((w, w0)))
                Dw_w0 = rhs[:n_masses*n_spectra]
                diffs = rhs[(n_masses+n_molecules)*n_spectra:]
            return logdot(Y, Dw_w0) - Dw_w0.sum() - lambda_ * w.sum() - theta * np.linalg.norm(diffs)**2

        # log-likelihood for the modified problem (variables w, w0, z0, z1, z2, u0, u1, u2)
        def LL_ADMM():
            return logdot(Y, z0) - z0.sum() - lambda_ * z1.sum() - theta * np.linalg.norm(z2)**2 \
                    - np.dot(u0, z0 - Dw_w0_estimate) \
                    - np.dot(u1, z1 - w_estimate) \
                    - np.dot(u2, z2 - diff_estimates) \
                    - rho/2 * np.linalg.norm(z0 - Dw_w0_estimate) ** 2 \
                    - rho/2 * np.linalg.norm(z1 - w_estimate) ** 2 \
                    - rho/2 * np.linalg.norm(z2 - diff_estimates) ** 2

        max_iter = 50
        rhs = None
        for i in range(max_iter):
            logging.info("w,w0 update")
            w_estimate, w0_estimate = w_w0_update()
            rhs_old = rhs
            rhs = w_w0_lasso.predict(A)
            Dw_w0_estimate = rhs[:n_masses*n_spectra]
            diff_estimates = rhs[(n_masses+n_molecules)*n_spectra:]
            #print "w,w0 update", LL(w_estimate, Dw_w0_estimate, diff_estimates)
            #print w_estimate.reshape((self.nrows, self.ncols))
            #print w0_estimate.reshape((self.nrows, self.ncols))
            logging.info("z0 update")
            #print "LL_ADMM after w updates:", LL_ADMM()
            z0_old, z1_old, z2_old = z0, z1, z2
            z0 = z0_update(Dw_w0_estimate, u0)
            #print np.linalg.norm(z0 - Dw_w0_estimate)
            #print "LL_ADMM after z0 update:", LL_ADMM()
            logging.info("z1 update")
            z1 = z1_update(w_estimate, u1)
            #print np.linalg.norm(z1 - w_estimate)
            #print "LL_ADMM after z1 update:", LL_ADMM()
            #print "z1 update", LL(z1, w0=w0_estimate)
            logging.info("z2 update")
            z2 = z2_update(diff_estimates, u2)

            #print np.linalg.norm(z2 - diff_estimates)
            #print "LL_ADMM after z2 update:", LL_ADMM()
            u_old = np.concatenate((u0, u1, u2))
            u0 += rho * (z0 - Dw_w0_estimate)
            u1 += rho * (z1 - w_estimate)
            u2 += rho * (z2 - diff_estimates)

            if rhs_old is not None:
                u = np.concatenate((u0, u1, u2))
                primal_diff = 1.0 / rho * np.linalg.norm(u - u_old)
                dual_diff = rho * np.linalg.norm(rhs - rhs_old)
                print primal_diff, dual_diff, primal_diff + dual_diff, LL(w_estimate, Dw_w0_estimate, diff_estimates)
            #print "LL_ADMM after u updates:", LL_ADMM()
            #print w_estimate.sum(), w0_estimate.sum(), z0.sum(), z1.sum(), z2.sum(), u0.sum(), u1.sum(), u2.sum()
            if i % 10 == 0 and i > 0:
                # TODO: exploit dual residuals for setting rho
                rho *= 2
                print "rho <-", rho
                print LL(w_estimate, Dw_w0_estimate, diff_estimates)
                print w_estimate.reshape((n_molecules, self.nrows, self.ncols), order='F').sum(axis=(1,2))
                w_w0_lasso = Lasso(alpha=lambda_/rho/A.shape[0], warm_start=True, fit_intercept=False, positive=True)
                z1_lasso = Lasso(alpha=lambda_/rho/z1.shape[0], fit_intercept=False, warm_start=True, positive=False)
                z2_ridge = ElasticNet(alpha=2*theta/rho/z2.shape[0], l1_ratio=0, warm_start=True, positive=False, fit_intercept=False)

        #print D.todense()
        #print (Y-Dw_w0_estimate).reshape((n_masses, n_spectra), order='F')
        print LL(w_estimate, Dw_w0_estimate, diff_estimates)
        print w_estimate.reshape((n_molecules, self.nrows, self.ncols), order='F').sum(axis=(1,2))
        #print w0_estimate.reshape((self.nrows, self.ncols), order='F')
        print self.sum_formulae


if __name__ == '__main__':
    import json
    import sys
    config = json.loads(open(sys.argv[1]).read())
    pipeline = ProbPipeline(config)
    pipeline.run()
