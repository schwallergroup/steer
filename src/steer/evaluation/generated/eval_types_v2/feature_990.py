"""Generated evaluation code for: Convergent synthesis via hydrazine and ketone fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentFischerIndole(BaseScoring):
    """
    Evaluates convergent synthesis routes that use Fischer indole synthesis
    to couple hydrazine and ketone fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        # Fischer indole reaction pattern: hydrazine + ketone -> indole
        self.hydrazine_pattern = Chem.MolFromSmarts("[NH2]-[NH]-c1ccccc1")  # phenylhydrazine
        self.ketone_pattern = Chem.MolFromSmarts("[#6]-[CX3](=O)-[#6]")  # ketone
        self.indole_pattern = Chem.MolFromSmarts("c1ccc2[nH]ccc2c1")  # indole core
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Fischer indole synthesis not found
        else:
            # Earlier Fischer indole synthesis is better for convergent strategy
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction is a Fischer indole synthesis between
        hydrazine and ketone fragments.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains indole
            if not product_mol.HasSubstructMatch(self.indole_pattern):
                return False
                
            # Parse reactants
            reactant_parts = reactant_smiles.split(".")
            if len(reactant_parts) < self.fragment_count:
                return False
                
            reactant_mols = []
            for r_smiles in reactant_parts:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactant_mols.append(mol)
                    
            if len(reactant_mols) < self.fragment_count:
                return False
                
            # Check for hydrazine and ketone fragments
            has_hydrazine = any(mol.HasSubstructMatch(self.hydrazine_pattern) 
                              for mol in reactant_mols)
            has_ketone = any(mol.HasSubstructMatch(self.ketone_pattern) 
                           for mol in reactant_mols)
            
            return has_hydrazine and has_ketone
            
        except Exception:
            return False
