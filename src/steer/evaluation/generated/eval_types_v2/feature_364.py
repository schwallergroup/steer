"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two major fragments are coupled
    via a specific reaction type at a target step.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["convergence_step"]
        self.expected_fragments = config["fragment_count"]
        self.coupling_type = config["coupling_reaction"]
        
        # Define SMARTS patterns for different coupling reactions
        self.coupling_patterns = {
            "williamson_ether": "[C:1][O:2][C:3]",  # Ether linkage
            "suzuki": "[c:1][c:2]",  # Aryl-aryl bond
            "click": "[C:1]1[N:2][N:3][N:4][C:5]1",  # Triazole from click
            "amide": "[C:1](=[O:2])[N:3]",  # Amide bond
            "reductive_amination": "[C:1][N:2][C:3]"  # C-N bond
        }

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent step not found
        # Perfect score if convergence happens at target step
        if abs(x - self.target_step) == 0:
            return 1
        # Penalty increases with distance from target step
        penalty = abs(x - self.target_step) * 0.2
        return max(0, 1 - penalty)

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step:
        1. Has the expected number of reactant fragments
        2. Forms the expected coupling bond type
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, product_smiles = mapped_rxn.split(">>")
        reactant_list = reactants_smiles.split(".")
        
        # Check if we have the expected number of fragments
        if len(reactant_list) != self.expected_fragments:
            return False
            
        # Check if reactants are substantial fragments (not just small molecules)
        substantial_fragments = 0
        for r_smi in reactant_list:
            mol = Chem.MolFromSmiles(r_smi)
            if mol and mol.GetNumAtoms() > 5:  # Consider >5 atoms as substantial
                substantial_fragments += 1
                
        if substantial_fragments != self.expected_fragments:
            return False
            
        # Check if the coupling reaction pattern is formed
        if self.coupling_type in self.coupling_patterns:
            pattern = self.coupling_patterns[self.coupling_type]
            pattern_mol = Chem.MolFromSmarts(pattern)
            product_mol = Chem.MolFromSmiles(product_smiles)
            
            if pattern_mol and product_mol:
                # Check if the coupling pattern is present in product
                if product_mol.HasSubstructMatch(pattern_mol):
                    # Verify this bond was actually formed (not present in reactants)
                    bond_formed = True
                    for r_smi in reactant_list:
                        r_mol = Chem.MolFromSmiles(r_smi)
                        if r_mol and r_mol.HasSubstructMatch(pattern_mol):
                            bond_formed = False
                            break
                    return bond_formed
                    
        return False
