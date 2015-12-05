from pyMS.pyisocalc import pyisocalc

from itertools import product
import os
import sys
import cPickle
import numpy as np
import scipy as sp
import scipy.sparse as ssp
import logging
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

class TVSolver(object):
    def __init__(self, nrows, ncols, pixel_indices, spectra, isotope_patterns, theta=1.0, lambda_=0.0):
        """ spectra: list of Spectrum objects
            isotope_patterns: list of tuples (mzs, intensities)
            theta: penalty on total variation
            lambda_: penalty on noise abundancies, zero by default
        """
        self.nrows = nrows
        self.ncols = ncols
        self.pixel_indices = pixel_indices
        self.spectra = spectra

        all_mzs = np.concatenate([pattern[0] for pattern in isotope_patterns])
        # append 'infinity' so that searchsorted always returns indices less than the length
        self.all_mz_int_indices = np.concatenate((np.unique(self.bin_numbers(all_mzs)), [np.iinfo(np.int32).max]))

        logging.info("computing Y matrix")
        self.Ysp = self.sparse_matrix_from_spectra([(s.mzs, s.intensities) for s in self.spectra])
        self.Y = np.array(self.Ysp.todense(order='F'))
        #self.Y *= 5e-1

        logging.info("computing D matrix")
        D = self.sparse_matrix_from_spectra(isotope_patterns, assume_presence=True, normalize=True)
        self.D = ssp.hstack((D, 1.0 / D.shape[0] * np.ones((D.shape[0], 1)))).tocsr()
        self.D_T = self.D.T.tocsr()

        self.n_masses, self.n_molecules = self.D.shape
        self.n_spectra = self.Y.shape[1]

        logging.info("computing GX, GY matrices")
        self.GX = self.compute_neighbor_matrix(1, 0)
        self.GY = self.compute_neighbor_matrix(0, 1)
        self._div_m = ssp.vstack((self.GX.T, self.GY.T))

        self.s = self.t = 0.3
        self.lambda_ = lambda_
        self.theta = theta
        self.alpha = 0.5
        self.eta = 0.95
        self.delta = 1.5

        self.scale = 1e2

        self.w_hat = self.D_T.dot(self.Y)
        #self.w_hat = np.zeros((self.n_molecules, self.n_spectra), dtype=np.float32)
        self.w1 = np.zeros(self.w_hat.shape, dtype=np.float32)
        self.y = np.zeros((self.n_masses, self.n_spectra), dtype=np.float32)
        self.p = np.zeros((self.n_molecules - 1, self.n_spectra * 2), dtype=np.float32)

        logging.info("initialized solver")

    def bin_numbers(self, mzs):
        # FIXME: this binning scheme is extremely dumb
        # value of 5.0 is for illustrative purposes only
        return (mzs * 5.0).astype(np.int32)

    def sparse_matrix_from_spectra(self, data, assume_presence=False, normalize=False):
        intensity_list = []
        row_list = []
        len_list = []
        for j, (mzs, intensities) in enumerate(data):
            int_mzs = self.bin_numbers(mzs)
            # all_mz_int_indices contain m/z values of known molecules    
            idx = self.all_mz_int_indices.searchsorted(int_mzs)
            if not assume_presence:
                known = np.where(self.all_mz_int_indices[idx] == int_mzs)[0]
                intensities = intensities[known]
                idx = idx[known]
                length = len(known)
            else: # short-cut for better performance
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
                              dtype=np.float32)

    def compute_neighbor_matrix(self, dx, dy):
        # builds incidence matrix for pixels
        # note that some pixels of the rectangle might be absent in the data
        neighbors_map = {}
        indices = -1 * np.ones((self.nrows, self.ncols), dtype=int)
        for i, s in enumerate(self.spectra):
            x, y = divmod(self.pixel_indices[i], self.ncols)
            indices[x, y] = s.index
        for x in xrange(self.nrows):
            for y in xrange(self.ncols):
                if indices[x, y] == -1:
                    continue
                if 0 <= x + dx < self.nrows and 0 <= y + dy < self.ncols:
                    idx = indices[x + dx, y + dy]
                    if idx != -1:
                        neighbors_map[indices[x, y]] = idx

        rows = np.concatenate([[i, j] for i, j in neighbors_map.items()])
        columns = np.concatenate([[i, i] for i, j in neighbors_map.items()])
        weights = np.concatenate([[1, -1] for i, j in neighbors_map.items()])
        return ssp.coo_matrix((weights, (rows, columns)),
                              shape=(self.n_spectra, self.n_spectra),
                              dtype=np.float32)

    def gradient(self, w):
        w_mol = w[:-1,:]
        return np.hstack((self.GX.T.dot(w_mol.T).T, self.GY.T.dot(w_mol.T).T))

    def divergence(self, p):
        result = -self._div_m.T.dot(p.T).T
        return np.vstack((result, np.zeros((1, result.shape[1]))))

    def gradient_norm(self, grad_w, p, q, r):
        @njit
        def gradient_norm_helper(grad_w, p, q, r):
            tmp_r = 0.0
            n, m = grad_w.shape
            # compute p-norms in m/z dimension,
            # then q-norms in gradient component dimension,
            # and then r-norms in pixel dimension
            for pixel_idx in range(m / 2):
                grad_x_idx = pixel_idx
                grad_y_idx = pixel_idx + m / 2
                tmp_px = 0.0
                tmp_py = 0.0
                if p == np.inf:
                    for mz_idx in range(n):
                        tmp_px = max(tmp_px, abs(grad_w[mz_idx, grad_x_idx]))
                        tmp_py = max(tmp_py, abs(grad_w[mz_idx, grad_y_idx]))
                else:
                    for mz_idx in range(n):
                        tmp_px += abs(grad_w[mz_idx, grad_x_idx]) ** p
                        tmp_py += abs(grad_w[mz_idx, grad_y_idx]) ** p
                    tmp_px **= 1.0 / p
                    tmp_py **= 1.0 / p

                tmp_q = 0.0
                if q == np.inf:
                    tmp_q = max(tmp_px, tmp_py)
                else:
                    tmp_q = (tmp_px ** q + tmp_py ** q) ** (1.0 / q)

                if r == np.inf:
                    tmp_r = max(tmp_r, tmp_q)
                else:
                    tmp_r += tmp_q ** r
            if r != np.inf:
                tmp_r **= 1.0 / r
            return tmp_r

        return gradient_norm_helper(grad_w, p, q, r)

    def total_variation(self, w):
        return self.gradient_norm(self.gradient(w), 1,1,1)

    def LL(self, w):
        z = self.D.dot(w)
        # self.lambda_  is currently disabled
        return (self.Y * np.log(z + 1e-32)).sum() - z.sum() - self.theta * self.total_variation(w) #- self.lambda_ * w[-1, :].sum()

    def run_single_iteration(self, calc_residuals=False):
        new_w_hat = self.w_hat - self.t * self.D_T.dot(self.y)
        new_w_hat += self.t * self.divergence(self.p)
        new_w_hat[new_w_hat < 0.0] = 0.0
        w1 = 2.0 * new_w_hat - self.w_hat
        tmp = self.y + self.s * self.D.dot(w1)

        # minimize Poisson distance between DW and Y (KL-divergence)
        new_y = 0.5 * (tmp + 1 - np.sqrt((tmp - 1) ** 2 + 4.0 * self.s * self.Y))

        # a faster way, but requires yet another library
        #import numexpr as ne
        #new_y = ne.evaluate('0.5 * (tmp + 1 - sqrt((tmp - 1) ** 2 + 4.0 * s * Y))',
        #        local_dict={'tmp': tmp, 's': self.s, 'Y': self.Y})

        # this is how to minimize Gaussian distance between DW and Y:
        #new_y = (tmp - self.s * self.Y) / (self.s + 1.0)

        tmp = self.p + self.s * self.gradient(w1)
        new_p = tmp

        # corresponds to l-2,2,2 norm of the gradient - leads to smoothing of edges
        #norm_p = np.linalg.norm(new_p)
        #if norm_p > self.theta:
        #    new_p /= (norm_p / self.theta)

        # l-1,1,1 results in projection to l-\infty ball
        new_p[new_p > self.theta] = self.theta
        new_p[new_p < -self.theta] = -self.theta

        if calc_residuals:
            # optional step that helps to accelerate convergence of the algorithm
            # it's not necessary to run it on every iteration
            delta_x = new_w_hat - self.w_hat
            delta_y = new_y - self.y
            delta_p = new_p - self.p

            p = np.linalg.norm(delta_x / self.t - self.D_T.dot(delta_y) + self.divergence(delta_p))
            d = (np.linalg.norm(delta_y / self.s - self.D.dot(delta_x)) ** 2 +
                 np.linalg.norm(delta_p / self.s - self.gradient(delta_x)) ** 2) ** 0.5
            if p > self.delta * self.scale * d:
                self.t /= (1.0 - self.alpha)
                self.s *= (1.0 - self.alpha)
                self.alpha *= self.eta
                #print "new t, s:", self.t, self.s, "|", p, d
            elif self.delta * p < self.scale * d:
                self.t *= (1.0 - self.alpha)
                self.s /= (1.0 - self.alpha)
                self.alpha *= self.eta
                #print "new t, s:", self.t, self.s, "|", p, d

            self.w_hat = new_w_hat
            self.y = new_y
            self.p = new_p

            return p, d
        else:
            self.w_hat = new_w_hat
            self.y = new_y
            self.p = new_p

            return None, None

    def kl_ratio(self, w):
        # ideally, the parameters should be set so that this ratio is close to 1
        # (Poisson discrepancy principle;
        #  an estimate for Gamma noise can also be found in literature)
        z = self.D.dot(w)
        return (z.sum() - (self.Y * np.log(z + 1e-32)).sum()) / (self.Ysp.nnz / 2.0)

    def get_image(self, w, molecule_index):
        # abundancy matrix for a given molecule
        abundancies = w.reshape((self.n_molecules, self.n_spectra), order='F')[molecule_index]
        img = np.zeros(self.nrows * self.ncols)
        img[self.pixel_indices] = abundancies
        return img.reshape((self.nrows, self.ncols))

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
        # copy-paste from pyIMS.ion_datacube
        bbox = []
        for ii in range(0,3):
            bbox.append(np.amin(self.coords[:,ii]))
            bbox.append(np.amax(self.coords[:,ii]))
        _coord = self.coords.round(5) - np.amin(self.coords, axis=0)
        step = np.zeros((3,1))
        for ii in range(0,3):
            step[ii] = np.mean(np.diff(np.unique(_coord[:,ii])))  
        # coordinate to pixels
        _coord /= np.reshape(step, (3,))
        _coord_max = np.amax(_coord,axis=0)
        self.ncols = int(_coord_max[1]+1)
        self.nrows = int(_coord_max[0]+1)
        self.pixel_indices = _coord[:,0] * self.ncols + _coord[:,1]
        self.pixel_indices = self.pixel_indices.astype(np.int32)

    def load_data(self):
        if self.data_file.endswith(".imzML"):
            spectra = readImzML(self.data_file)
        else:
            raise "the input format is unsupported"

        self.spectra = list(spectra)

        self.coords = np.zeros((len(self.spectra), 3))
        for i, sp in enumerate(self.spectra):
            self.coords[i, :] = sp.coords
        self._calculate_dimensions()

    def get_solver(self, **kwargs):
        isotope_patterns = [self.mz_list[f][a] for f in self.sum_formulae for a in self.adducts]
        return TVSolver(self.nrows, self.ncols, self.pixel_indices, self.spectra, isotope_patterns, **kwargs)
