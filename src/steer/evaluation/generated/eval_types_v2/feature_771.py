"""Generated evaluation code for: Convergent synthesis via two key fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if the route couples
    a specified number of key fragments at a target depth.
    """
    
    def __init__(self, config: Dict):
        self.target_fragment_count = config["fragment_count"]
        self.target_coupling_depth = config["coupling_depth"]
        self.min_fragment_size = config.get("min_fragment_size", 6)  # minimum atoms per fragment
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Perfect score if coupling happens at target depth
            depth_penalty = abs(x - (self.target_coupling_depth / 10.0))
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of the target number of fragments
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1].split(".")
        
        # Filter reactants to only include substantial fragments (not reagents/catalysts)
        substantial_reactants = []
        for reactant_smiles in reactants_smiles:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.GetNumHeavyAtoms() >= self.min_fragment_size:
                    substantial_reactants.append(reactant_smiles)
            except:
                continue
        
        # Check if we have the target number of fragments coupling
        if len(substantial_reactants) != self.target_fragment_count:
            return False
        
        # Verify this is a true coupling reaction (fragments combine to form product)
        return self._is_coupling_reaction(product_smiles, substantial_reactants)
    
    def _is_coupling_reaction(self, product_smiles: str, reactant_smiles_list: List[str]) -> bool:
        """
        Verify that the reactants are actually coupling to form the product
        by checking that key substructures from each reactant appear in the product
        """
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
            
            reactant_mols = []
            for smiles in reactant_smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    reactant_mols.append(mol)
            
            if len(reactant_mols) != len(reactant_smiles_list):
                return False
            
            # Check that each reactant contributes a substantial portion to the product
            product_atoms = product_mol.GetNumHeavyAtoms()
            total_reactant_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactant_mols)
            
            # Coupling reaction should have combined atom count close to product
            # (allowing for small differences due to leaving groups, etc.)
            atom_ratio = total_reactant_atoms / product_atoms if product_atoms > 0 else 0
            
            return 0.7 <= atom_ratio <= 1.3  # Allow 30% variance for functional group changes
            
        except:
            return False
