"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting coupling reactions
    that combine major fragments at a specific depth in the synthesis tree.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction_type = config.get("coupling_reaction_type", "esterification")
        
        # Define SMARTS patterns for different coupling reactions
        self.coupling_patterns = {
            "esterification": ["[C:1](=[O:2])[OH:3].[OH:4]", "[C:1](=[O:2])[Cl:3].[OH:4]"],
            "amidation": ["[C:1](=[O:2])[OH:3].[NH2:4]", "[C:1](=[O:2])[Cl:3].[NH2:4]"],
            "suzuki": ["[c:1][B:2]([OH:3])[OH:4].[c:5][Br:6]", "[c:1][B:2]([OH:3])[OH:4].[c:5][I:6]"],
            "click": ["[C:1]#[C:2].[N:3]=[N+:4]=[N-:5]"],
            "olefin_metathesis": ["[C:1]=[C:2].[C:3]=[C:4]"]
        }
    
    def route_scoring(self, x) -> float:
        """
        Score based on convergent coupling depth.
        Earlier convergent coupling (lower depth) gets higher score.
        """
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Earlier coupling is better - score decreases with depth
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Check if we have the expected number of fragments
            if len(reactants) != self.fragment_count:
                return False
            
            # Convert to RDKit molecules
            reactant_mols = []
            for r_smi in reactants:
                mol = Chem.MolFromSmiles(r_smi)
                if mol is None:
                    return False
                reactant_mols.append(mol)
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
            
            # Check if this is a convergent step by verifying:
            # 1. Multiple substantial fragments are being combined
            # 2. The reaction matches the specified coupling type
            if self._is_convergent_coupling(reactant_mols, product_mol):
                return self._matches_coupling_type(reactants_smiles, product_smiles)
                
        except Exception:
            return False
        
        return False
    
    def _is_convergent_coupling(self, reactant_mols, product_mol) -> bool:
        """
        Check if this represents a true convergent coupling between substantial fragments.
        """
        # Each reactant should be a substantial fragment (>= 5 heavy atoms)
        min_fragment_size = 5
        
        for mol in reactant_mols:
            heavy_atom_count = mol.GetNumHeavyAtoms()
            if heavy_atom_count < min_fragment_size:
                return False
        
        # Product should contain structural elements from both fragments
        product_heavy_atoms = product_mol.GetNumHeavyAtoms()
        total_reactant_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactant_mols)
        
        # Allow for small differences due to coupling (loss of small groups like H2O, HCl)
        if abs(product_heavy_atoms - total_reactant_atoms) > 2:
            return False
            
        return True
    
    def _matches_coupling_type(self, reactants_smiles, product_smiles) -> bool:
        """
        Check if the reaction matches the specified coupling reaction type.
        """
        if self.coupling_reaction_type not in self.coupling_patterns:
            # If coupling type not defined, accept any convergent reaction
            return True
        
        patterns = self.coupling_patterns[self.coupling_reaction_type]
        
        for pattern in patterns:
            if "." in pattern:
                # Multi-component pattern
                pattern_parts = pattern.split(".")
                reactant_parts = reactants_smiles.split(".")
                
                if len(pattern_parts) == len(reactant_parts):
                    matches = 0
                    for r_smi in reactant_parts:
                        r_mol = Chem.MolFromSmiles(r_smi)
                        if r_mol:
                            for p_pattern in pattern_parts:
                                p_mol = Chem.MolFromSmarts(p_pattern)
                                if p_mol and r_mol.HasSubstructMatch(p_mol):
                                    matches += 1
                                    break
                    
                    if matches == len(pattern_parts):
                        return True
            else:
                # Single pattern - check against combined reactants
                combined_mol = Chem.MolFromSmiles(reactants_smiles.replace(".", ""))
                if combined_mol:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and combined_mol.HasSubstructMatch(pattern_mol):
                        return True
        
        return False
