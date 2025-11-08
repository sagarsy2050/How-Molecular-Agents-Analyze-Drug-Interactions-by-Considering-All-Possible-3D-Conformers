from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
import numpy as np

def generate_conformers(mol, num_confs=30):
    if not mol:
        return mol, []
    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.75
    params.numThreads = 0
    cids = rdDistGeom.EmbedMultipleConfs(mol, num_confs, params)

    if len(cids) == 0:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)

    if len(cids) == 0:
        return mol, []

    # FINAL FIX: Correct MMFF call with mmffProp=
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        # ← THIS IS THE CORRECT SIGNATURE
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffProp=mmff_props, maxIters=1000, numThreads=0)
    except Exception as e:
        print(f"MMFF optimization skipped (continuing with UFF): {e}")

    # UFF final energies
    energies = []
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol)
        for cid in range(mol.GetNumConformers()):
            ff.Initialize()
            ff.Minimize(confs=cid, maxIts=1000)
            energies.append(ff.CalcEnergy())
    except Exception:
        energies = list(range(len(cids)))

    sorted_ids = np.argsort(energies)[:5]
    return mol, sorted_ids.tolist()