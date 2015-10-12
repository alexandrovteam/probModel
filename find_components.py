import numpy as np
import scipy.sparse as ssp
from pyMS.pyisocalc import pyisocalc
sum_formulae = [l.strip() for l in open("formulae.txt")]

adducts = ['H', 'Na', 'K']
patterns = {}
import os
import cPickle
if os.path.isfile("patterns.pkl"):
    patterns = cPickle.load(open("patterns.pkl"))
else:
    print "generating patterns..." 
    for n, sum_formula in enumerate(sum_formulae):   
        for adduct in adducts:
            isotope_ms = pyisocalc.isodist(sum_formula + adduct,
                                           plot=False,
                                           sigma=0.01,
                                           charges=1,
                                           resolution=200000,
                                           do_centroid=True)
            if not sum_formula in patterns:
                 patterns[sum_formula] = {}
            patterns[sum_formula][adduct] = isotope_ms.get_spectrum(source='centroids')
    with open('patterns.pkl', 'w') as f:
        cPickle.dump(patterns, f)

formulas = [k + '+' + a for k in patterns for a in patterns[k]]
masses = [patterns[k][a][0] for k in patterns for a in patterns[k]]
all_masses = np.concatenate(masses)
order = all_masses.argsort()
all_masses = all_masses[order]
mol_indices = np.repeat(np.arange(len(masses)), map(len, masses))[order]
mass_diffs = np.diff(all_masses)

def find_components(ppm, targets):
    small_diff_indices = np.where(mass_diffs < ppm*all_masses[1:])[0]
    pairs = sorted(set(zip(mol_indices[small_diff_indices], mol_indices[small_diff_indices+1])))
    connections = ssp.coo_matrix((np.repeat(1, len(pairs)), zip(*pairs)), shape=(len(masses), len(masses)), dtype=int)
    ncomponents, coloring = ssp.csgraph.connected_components(connections)
    component_sizes = np.sort(np.bincount(coloring))[::-1]
    target_components = {}
    for target in targets:
        target_index = formulas.index(target)
        target_color = coloring[target_index]
        if np.sum(coloring == coloring[target_index]) > 50:
            continue
        if target_color not in target_components:
            target_components[target_color] = [target]
        else:
            target_components[target_color].append(target)
    d = {color: items for color, items in target_components.items() if len(items) > 2}
    print d
    for color in d:
        print color
        for f in sorted(set(formulas[i].split('+')[0] for i in np.where(coloring == color)[0])):
            print f
    #print len(component_sizes)
    #print component_sizes
    
targets = open("decoy_results/decoy_pass_results_new.txt").readlines()[1:]
targets = [f+'+'+a for f, a in [l.split(",")[:2] for l in targets]]
for ppm in xrange(1, 11):
    ppm /= 1e6
    print ppm
    find_components(ppm, targets)
