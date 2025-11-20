"""Generated evaluation code for: Multi-component heterocycle core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiComponentPyridoPyrimidineFormation(BaseScoring):
    """
    Evaluates routes for multi-component formation of pyrido[4,3-d]pyrimidine core.
    Checks if the target heterocycle is formed via a multi-component reaction
    involving exactly 3 components in a single step.
    """
    
    def __init__(self, config: Dict):
        self.component_count = config["parameters"].get("component_count", 3)
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
        
        # SMARTS pattern for pyrido[4,3-d]pyrimidine core
        self.pyridopyrimidine_pattern = "c1cnc2c(n1)cncc2"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Multi-component formation doesn't happen
        else:
            # Earlier formation (lower depth) is better for core formation
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents multi-component pyrido[4,3-d]pyrimidine formation
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Check if product contains pyrido[4,3-d]pyrimidine core
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            pattern_mol = Chem.MolFromSmarts(self.pyridopyrimidine_pattern)
            if not product_mol.HasSubstructMatch(pattern_mol):
                return False
            
            # Check if reaction involves exactly the specified number of components
            reactant_list = reactants_smiles.split(".")
            if len(reactant_list) != self.component_count:
                return False
                
            # Verify that none of the reactants already contain the target core
            for reactant_smiles in reactant_list:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol and reactant_mol.HasSubstructMatch(pattern_mol):
                    return False  # Core already exists, not a formation reaction
                    
            # Additional check: ensure this is actually forming the heterocycle
            # by verifying that key nitrogen atoms are being connected
            return self._verify_ring_formation(product_mol, reactant_list)
            
        except Exception:
            return False
    
    def _verify_ring_formation(self, product_mol, reactant_smiles_list) -> bool:
        """
        Verify that the pyrido[4,3-d]pyrimidine core is actually being formed
        by checking that nitrogen atoms from different reactants are being connected
        """
        try:
            reactant_mols = []
            for smiles in reactant_smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    reactant_mols.append(mol)
            
            if len(reactant_mols) != self.component_count:
                return False
                
            # Check that at least two reactants contain nitrogen atoms
            # (required for pyrido[4,3-d]pyrimidine formation)
            n_containing_reactants = 0
            for mol in reactant_mols:
                if any(atom.GetSymbol() == 'N' for atom in mol.GetAtoms()):
                    n_containing_reactants += 1
                    
            return n_containing_reactants >= 2
            
        except Exception:
            return False
