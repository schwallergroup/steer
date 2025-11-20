"""Generated evaluation code for: Orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates the use of orthogonal protecting group strategies in synthesis routes.
    Checks for the presence of specified protecting groups and whether they are used
    in an orthogonal manner (can be removed under different conditions).
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", [])
        self.orthogonal = config.get("orthogonal", True)
        self.functional_groups = config.get("functional_groups", [])
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[NH,N][C](=O)OC(C)(C)C",  # tert-butoxycarbonyl
            "PNB": "[NH,N][C](=O)Oc1ccc(cc1)[N+](=O)[O-]",  # para-nitrobenzyl
            "Tce": "[NH,N][C](=O)OCC(Cl)(Cl)Cl",  # trichloroethyl
            "Cbz": "[NH,N][C](=O)OCc1ccccc1",  # benzyloxycarbonyl
            "Fmoc": "[NH,N][C](=O)OCC1c2ccccc2-c2ccccc21",  # fluorenylmethoxycarbonyl
            "TBS": "[OH,O][Si](C)(C)C(C)(C)C",  # tert-butyldimethylsilyl
            "Ac": "[NH,N,OH,O][C](=O)C",  # acetyl
            "Bn": "[NH,N,OH,O]Cc1ccccc1"  # benzyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        detected_pgs = []
        
        for rxn in reactions:
            for pg_name in self.protecting_groups:
                if self.detect_protecting_group(rxn, pg_name):
                    if pg_name not in detected_pgs:
                        detected_pgs.append(pg_name)
        
        # Check if we found the expected protecting groups
        found_all_pgs = all(pg in detected_pgs for pg in self.protecting_groups)
        
        # If orthogonal strategy is required, check that we have multiple different PGs
        if self.orthogonal:
            orthogonal_condition = len(set(detected_pgs)) >= 2 and found_all_pgs
        else:
            orthogonal_condition = found_all_pgs
        
        return orthogonal_condition, len(reactions)
    
    def detect_protecting_group(self, rxn, pg_name):
        """
        Detect if a protecting group is being installed or removed in a reaction.
        """
        if pg_name not in self.pg_patterns:
            return False
        
        pattern = self.pg_patterns[pg_name]
        mol_pattern = Chem.MolFromSmarts(pattern)
        
        if mol_pattern is None:
            return False
        
        # Parse reaction SMILES
        rxn_smiles = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
        if ">>" not in rxn_smiles:
            return False
        
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Check reactants
        reactant_has_pg = False
        for r_smiles in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol and mol.HasSubstructMatch(mol_pattern):
                reactant_has_pg = True
                break
        
        # Check products
        product_has_pg = False
        for p_smiles in products_smiles.split("."):
            mol = Chem.MolFromSmiles(p_smiles)
            if mol and mol.HasSubstructMatch(mol_pattern):
                product_has_pg = True
                break
        
        # Protection (PG appears in product) or deprotection (PG disappears from reactant)
        return (not reactant_has_pg and product_has_pg) or (reactant_has_pg and not product_has_pg)
    
    def route_scoring(self, x):
        """
        Score based on whether orthogonal protecting group strategy is used.
        Higher scores for successful implementation of the strategy.
        """
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 10  # Strategy successfully implemented
