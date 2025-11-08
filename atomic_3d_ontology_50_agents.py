#!/usr/bin/env python3
import os
import sys
import argparse
import json   # ← FIXED: ADDED HERE
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from conformers.generator import generate_conformers
from ontology.builder import build_ontology, get_atomic_features
from agents import AGENTS, run_agent

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data.csv")
    p.add_argument("--out", default="atomic_3d_ontology.owl")
    p.add_argument("--ollama", action="store_true")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--confs", type=int, default=30)
    
    if "ipykernel" in sys.modules:
        return argparse.Namespace(csv="data.csv", out="atomic_3d_ontology.owl", ollama=True, model="llama3.2", confs=30)
    return p.parse_args()

def main():
    args = get_args()
    
    if not os.path.exists(args.csv):
        pd.DataFrame({
            "SMILES": ["c1ccccc1", "FCc1ccccc1", "CCO", "c1ccncc1", "CC(=O)O"],
            "CLASS": ["Active", "Active", "Inactive", "Active", "Inactive"]
        }).to_csv(args.csv, index=False)
        print(f"Sample data created: {args.csv}")

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} molecules")

    from rdkit import Chem
    mol = Chem.MolFromSmiles(df.iloc[0]["SMILES"])
    mol_h, conf_ids = generate_conformers(mol, args.confs)
    if not conf_ids:
        print("No conformers generated!")
        return

    feats = get_atomic_features(mol_h, conf_ids[0])
    context = f"SMILES: {df.iloc[0]['SMILES']}\nActivity: {df.iloc[0]['CLASS']}\n{json.dumps(feats, indent=2)[:8000]}"

    print(f"\nLAUNCHING {len(AGENTS)} ATOMIC AGENTS...")
    for agent in tqdm(AGENTS):
        run_agent(agent, context, args.ollama, args.model)

    build_ontology(df, args.out)

    print("\n" + "="*100)
    print("50+ ATOMIC AGENTS + 3D ONTOLOGY → TOTAL SUCCESS")
    print(f"OWL saved: {args.out}")
    print("REAL LLM INSIGHTS WITH --ollama")
    print("="*100)

if __name__ == "__main__":
    main()