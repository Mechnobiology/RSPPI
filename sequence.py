import os
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB import PDBIO
from Bio.PDB.DSSP import DSSP


####### seqlen ##################################
# measure the length of both chain,
# if both are longer than the criteria,
# return True and the sequence of both chains
# and split the model to two structure and write pdb
# inputs: path, pdbid, criteria
# path: the path to the pdbfile
# pdbid: the pdb id
# criterial: the length criteria of sequence
# if either is shorter than the criteria,
# return False and "Nan"
###################################################

def seqlen(path, save_path, pdbid, criteria):
    parser = PDBParser(QUIET=True)
    io = PDBIO()
    filename = path + pdbid + '.pdb'
    struc = parser.get_structure(pdbid, filename)
    model = struc.get_list()[0]
    chains = model.get_list()

    if len(chains) > 2:
        return None, pdbid + "has more than two chains"
    elif len(chains) == 2:
        if len(chains[0].get_list()) < criteria or len(chains[1].get_list()) < criteria:
            return False, "Nan"
        else:
            sequence = []
            for chain in chains:
                residues = chain.get_list()
                seq = ''
                io.set_structure(chain)
                io.save(save_path + pdbid + "." + chain.get_id() + ".pdb")
                for residue in residues:
                    seq = seq + seq1(residue.get_resname())

                sequence.append(seq)

    else:
        return None, pdbid + "has less than two chains"

    return True, sequence


###############################################
# after split pdb files to two chains files,
# find out exposed amino acid and return seqs,
# inputs: path, pdbid, criteria
# path: the path of chains file
# pdbid: chains id
# criteria: RSA threshold
###############################################

def find_out_exposed_seq(path, pdbid, criteria):
    p = PDBParser(QUIET=True)
    in_file = path + '/' + pdbid + '.pdb'
    structure = p.get_structure(pdbid, in_file)
    model = structure[0]
    chain = model.get_list()
    try:
        dssp = DSSP(model, in_file)
    except Exception:
        return "NAN"
    a_key = list(dssp.keys())

    exposed_id_list = []
    key_list = []
    for key in a_key:
        if key[1][2] == " " and dssp[key][3] != "NA" and dssp[key][3] >= criteria:
            exposed_id_list.append(key[1][1])
            key_list.append(key[1])

    seq = ''
    for key in key_list:
        try:
            res = chain[0][key]
            seq = seq + seq1(res.get_resname())
        except KeyError:
            found = False

    chain_exposed_seq = {}
    chain_exposed_seq[pdbid] = seq

    return chain_exposed_seq


