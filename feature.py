from Bio.PDB.PDBParser import PDBParser
from math import sqrt
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from biopandas.pdb import PandasPdb
import pickle
import numpy as np
import pandas as pd

dict_hydrophobe = {'R':-7.5, 'K':-4.6, 'D':-3.0, 'Q':-2.9, 'N':-2.7, 'E':-2.6, 'H':-1.7, 'S':-1.1, 'T':-0.8, 'P':-0.3,
                  'Y':0.1, 'C':0.2, 'G':0.7, 'A':1.0, 'M':1.1, 'W':1.5, 'L':2.2, 'V':2.3, 'F':2.5, 'I':3.1}

dict_charge = {'R':10.8, 'K':9.7, 'D':2.8, 'Q':5.7, 'N':5.4, 'E':3.2,
                   'H':7.6, 'S':5.7, 'T':5.9, 'P':6.5,'Y':5.7, 'C':5.1,
                   'G':6.0, 'A':6.0, 'M':5.7, 'W':6.0, 'L':6.0, 'V':6.0, 'F':5.5, 'I':5.9}

#normalization
#dt: a matrix
def feature_normalize(dt):
    maxval = np.max(dt)
    minval = np.min(dt)
    val = (dt-minval) / (maxval-minval)
    return val

#calculate two atom's distance
#p1/p2: a list contain the x, y, z coordinate of a atom
def calc_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    distsq = pow(dx, 2) + pow(dy, 2) + pow(dz, 2)
    distance = sqrt(distsq)
    return distance

########### feature_map ########################
# calculate the min heavy atom(non-hydrogen atoms) distance map, hydrophobic map and charge map of a protein,
#inputs: path, pdbid, criteria
#path: the path to the pdbfile
#pdbid: the pdb id
#criteria: the length criteria of sequence
#################################################
def feature_map(path, pdbid, criteria):
    p = PDBParser(QUIET=True)
    in_file = path + "/" + pdbid + ".pdb"
    structure = p.get_structure(pdbid, in_file)
    model = structure[0]
    chain = model.get_list()
    residue = chain[0]
    dssp = DSSP(model, in_file)
    a_key = list(dssp.keys())

    exposed_id_list = []
    key_list = []

    for key in a_key:
        if key[1][2] == " " and dssp[key][3] != "NA" and dssp[key][3] >= criteria:
            key_list.append(key[1])

    seq = ''
    for key in key_list:
        try:
            res = chain[0][key]
            exposed_id_list.append(key[1])
            seq = seq + seq1(res.get_resname())
        except KeyError:
            found = False


    ppdb = PandasPdb()
    ppdb.read_pdb(in_file)
    number = len(exposed_id_list)
    answer = np.ones((number, number), np.float64)
    for i in range(number):
        coord1 = list(residue[exposed_id_list[i]]['CA'].get_vector())
        for j in range(i + 1, number):
            coord2 = list(residue[exposed_id_list[j]]['CA'].get_vector())
            dist = calc_dist(coord1, coord2)
            if dist > 30:
                answer[i, j] = 0
                answer[j, i] = 0
            else:
                p1 = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H'][
                    ppdb.df['ATOM']['residue_number'] == exposed_id_list[i]]
                p2 = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H'][
                    ppdb.df['ATOM']['residue_number'] == exposed_id_list[j]]
                p1_coord = []
                p2_coord = []
                for indexs1 in p1.index:
                    rowdata1 = p1.loc[indexs1].values[11:14]
                    rowdata1 = rowdata1.tolist()
                    p1_coord.append(rowdata1)
                for indexs2 in p2.index:
                    rowdata2 = p2.loc[indexs2].values[11:14]
                    rowdata2 = rowdata2.tolist()
                    p2_coord.append(rowdata2)
                distmin = 30
                for m in range(len(p1_coord)):
                    for n in range(len(p2_coord)):
                        atom_dist = calc_dist(p1_coord[m], p2_coord[n])
                        if atom_dist < distmin:
                            distmin = atom_dist

                if distmin > 14.0:
                    answer[i, j] = 0
                    answer[j, i] = 0
                else:
                    d0 = 4.0
                    cut_off = max(d0, distmin)
                    sij = 2 / (1 + cut_off / d0)
                    answer[i, j] = sij
                    answer[j, i] = sij

    hy_mat = np.zeros((number, number), np.float64)
    hy_df = pd.DataFrame((hy_mat), index=list(seq), columns=list(seq))
    for i in range(len(seq)):
        for j in range(len(seq)):
            # if hy_df.iloc[i, j] != 0:
            hy_df.iloc[i, j] = 20 - abs(
                (dict_hydrophobe[hy_df.index[i]] - dict_hydrophobe[hy_df.columns[j]]) * 19 / 10.6)

    charge_mat = np.zeros((number, number), np.float64)
    charge_df = pd.DataFrame((charge_mat), index=list(seq), columns=list(seq))
    for i in range(len(seq)):
        for j in range(len(seq)):
            # if charge_df.iloc[i, j] != 0:
            charge_df.iloc[i, j] = 11 - (dict_charge[charge_df.index[i]] - 7) * (
                    dict_charge[charge_df.index[j]] - 7) * 19 / 33.8

    distmap = {}
    distmap[pdbid] = answer
    dictf = {}
    HCM = feature_normalize(hy_df.to_numpy())
    CCM = feature_normalize(charge_df.to_numpy())
    feature_3D_array = np.array([answer, HCM, CCM])
    dictf[pdbid] = feature_3D_array
    return dictf

