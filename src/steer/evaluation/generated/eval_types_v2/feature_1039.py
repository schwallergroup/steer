"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if a specific coupling reaction
    occurs at a target depth with the required number of fragments.
    
    Convergent synthesis involves joining two or more complex fragments in a single step,
    typically more efficient than linear assembly approaches.
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config["parameters"]["coupling_depth"]
        self.required_fragments = config["parameters"]["fragment_count"]
        self.coupling_reaction = config["parameters"].get("coupling_reaction", "")
        
        # Define SMARTS patterns for common coupling reactions
        self.coupling_patterns = {
            "Williamson ether synthesis": "[C:1][O:2][C:3]",
            "Suzuki coupling": "[c:1][c:2]",  # Simplified Ar-Ar bond
            "Heck reaction": "[C:1]=[C:2][c:3]",  # Alkene-aryl
            "Click reaction": "[C:1]1[N:2][N:3][N:4][C:5]1",  # Triazole formation
            "amide coupling": "[C:1](=[O:2])[N:3]",
            "ester coupling": "[C:1](=[O:2])[O:3][C:4]"
        }
    
    def route_scoring(self, x) -> float:
        """Convert depth result to score (0-10 scale)"""
        if x < 0:
            return 0  # Condition never met
        
        # Perfect score if coupling happens at target depth
        if abs(x - self.target_depth) < 0.01:
            return 10
        
        # Penalize deviation from target depth
        depth_penalty = abs(x - self.target_depth) * 2
        return max(0, 10 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents the desired convergent coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            reactants = react_smiles.split(".")
            
            # Check if we have the required number of fragments
            if len(reactants) != self.required_fragments:
                return False
            
            # If specific coupling reaction specified, check for its pattern
            if self.coupling_reaction and self.coupling_reaction in self.coupling_patterns:
                pattern = self.coupling_patterns[self.coupling_reaction]
                product_mol = Chem.MolFromSmiles(prod_smiles)
                
                if product_mol is None:
                    return False
                
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol is None:
                    return False
                
                # Check if the coupling pattern is present in product
                if not product_mol.HasSubstructMatch(pattern_mol):
                    return False
                
                # Verify the pattern atoms come from different reactants
                return self._verify_convergent_coupling(prod_smiles, reactants, pattern)
            
            # If no specific reaction, just check fragment count and complexity
            return self._check_fragment_complexity(reactants)
            
        except Exception:
            return False
    
    def _verify_convergent_coupling(self, product_smiles: str, reactant_smiles: List[str], pattern: str) -> bool:
        """Verify that the coupling bond connects atoms from different reactants"""
        try:
            prod_mol = Chem.MolFromSmiles(product_smiles)
            pattern_mol = Chem.MolFromSmarts(pattern)
            
            if not prod_mol or not pattern_mol:
                return False
            
            # Find the pattern match in product
            matches = prod_mol.GetSubstructMatches(pattern_mol)
            if not matches:
                return False
            
            # Get atom map numbers for the coupling atoms (first two atoms in pattern)
            match = matches[0]
            coupling_atoms = [prod_mol.GetAtomWithIdx(match[0]).GetAtomMapNum(),
                            prod_mol.GetAtomWithIdx(match[1]).GetAtomMapNum()]
            
            # Check that these atoms come from different reactants
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactant_smiles]
            atom_locations = []
            
            for coupling_atom in coupling_atoms:
                for i, react_mol in enumerate(reactant_mols):
                    if react_mol and any(a.GetAtomMapNum() == coupling_atom for a in react_mol.GetAtoms()):
                        atom_locations.append(i)
                        break
            
            # True if coupling atoms come from different reactants
            return len(set(atom_locations)) == len(coupling_atoms) == 2
            
        except Exception:
            return False
    
    def _check_fragment_complexity(self, reactant_smiles: List[str]) -> bool:
        """Check if reactants are sufficiently complex to be considered fragments"""
        min_heavy_atoms = 5  # Minimum size to be considered a "complex fragment"
        
        try:
            for smi in reactant_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol and mol.GetNumHeavyAtoms() < min_heavy_atoms:
                    return False
            return True
        except Exception:
            return False
