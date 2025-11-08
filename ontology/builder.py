from owlready2 import get_ontology, Thing, DataProperty, ObjectProperty
from rdkit import Chem
from rdkit.Chem import AllChem  # ← FIXED: WAS MISSING
import numpy as np
import json
from conformers.generator import generate_conformers

def get_atomic_features(mol, conf_id):
    try:
        AllChem.ComputeGasteigerCharges(mol)  # ← NOW WORKS
    except:
        pass  # some molecules fail, we ignore

    energy = 99999.0
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        energy = ff.CalcEnergy()
    except:
        pass

    def dipole(conf):
        d = np.zeros(3)
        for a in mol.GetAtoms():
            q = a.GetDoubleProp("_GasteigerCharge") if a.HasProp("_GasteigerCharge") else 0.0
            pos = np.array(conf.GetAtomPosition(a.GetIdx()))
            d += q * pos
        return float(np.linalg.norm(d))

    def pmi(conf):
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        centered = coords - coords.mean(0)
        inertia = np.dot(centered.T, centered)
        eig = sorted(np.linalg.eigvals(inertia).real)
        return eig[0], eig[1], eig[2]

    conf = mol.GetConformer(conf_id)
    I1, I2, I3 = pmi(conf)
    asph = (I3 - I1) / I3 if I3 > 1e-5 else 0.0

    feats = {
        "conformer_energy": float(energy),
        "dipole_moment": dipole(conf),
        "asphericity": asph,
        "I1": I1, "I2": I2, "I3": I3,
        "atoms": []
    }

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        charge = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        feats["atoms"].append({
            "idx": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "charge": float(charge),
            "x": float(pos.x), "y": float(pos.y), "z": float(pos.z),
            "hybridization": str(atom.GetHybridization()),
            "aromatic": atom.GetIsAromatic(),
            "in_ring": atom.IsInRing(),
        })
    return feats

def build_ontology(df, path="atomic_3d_ontology.owl"):
    onto = get_ontology("http://atomic3d.org/#")
    with onto:
        class Molecule(Thing): pass
        class Conformer(Thing): pass
        class Atom(Thing): pass
        class has_conformer(Molecule >> Conformer, ObjectProperty): pass
        class has_atom(Conformer >> Atom, ObjectProperty): pass
        class energy(Conformer >> float, DataProperty): pass
        class dipole(Conformer >> float, DataProperty): pass
        class charge(Atom >> float, DataProperty): pass
        class x(Atom >> float, DataProperty): pass
        class y(Atom >> float, DataProperty): pass
        class z(Atom >> float, DataProperty): pass

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if not mol: 
            print(f"Invalid SMILES: {row['SMILES']}")
            continue
        m = onto.Molecule(f"M_{idx}")
        mol_h, conf_ids = generate_conformers(mol, 30)
        if not conf_ids: 
            print(f"No conformers for {row['SMILES']}")
            continue

        for cid in conf_ids[:2]:
            c = onto.Conformer(f"C_{idx}_{cid}")
            feats = get_atomic_features(mol_h, cid)
            c.energy = [feats["conformer_energy"]]
            c.dipole = [feats["dipole_moment"]]
            m.has_conformer.append(c)

            for a in feats["atoms"]:
                atom = onto.Atom(f"A_{idx}_{cid}_{a['idx']}")
                atom.charge = [a["charge"]]
                atom.x = [a["x"]]
                atom.y = [a["y"]]
                atom.z = [a["z"]]
                c.has_atom.append(atom)

    onto.save(file=path, format="rdfxml")
    print(f"ONTOLOGY SAVED → {path}")