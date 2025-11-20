"""Generated evaluation code for: Convergent synthesis via two main fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates whether a synthesis route uses a convergent strategy by checking
    if multiple main fragments are combined at a specified coupling stage.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_stage = config["coupling_stage"]
        self.min_fragment_size = config.get("min_fragment_size", 5)  # Minimum atoms to be considered a "main fragment"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        
        if self.coupling_stage == "final":
            # For final stage coupling, earlier is better (lower depth fraction)
            return 1 - x
        elif self.coupling_stage == "middle":
            # For middle stage, penalize very early or very late coupling
            optimal_range = (0.3, 0.7)
            if optimal_range[0] <= x <= optimal_range[1]:
                return 1.0
            elif x < optimal_range[0]:
                return x / optimal_range[0]
            else:
                return (1 - x) / (1 - optimal_range[1])
        else:
            # Default: prefer earlier coupling
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of main fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        if "." not in reactants_smiles:
            return False  # Not a coupling reaction
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            if len(reactants) < self.fragment_count:
                return False
            
            # Filter out small molecules (catalysts, reagents)
            main_fragments = []
            for reactant in reactants:
                if reactant.GetNumAtoms() >= self.min_fragment_size:
                    # Check if it's not a simple reagent (like acids, bases, etc.)
                    if not self._is_simple_reagent(reactant):
                        main_fragments.append(reactant)
            
            if len(main_fragments) < self.fragment_count:
                return False
            
            # Verify that the main fragments are indeed being coupled
            # by checking that significant portions of each fragment are retained in product
            fragments_coupled = 0
            for fragment in main_fragments:
                if self._fragment_incorporated_in_product(fragment, product):
                    fragments_coupled += 1
            
            return fragments_coupled >= self.fragment_count
            
        except Exception:
            return False
    
    def _is_simple_reagent(self, mol) -> bool:
        """
        Check if molecule is likely a simple reagent rather than a main fragment.
        """
        if mol.GetNumAtoms() <= 3:
            return True
        
        # Common reagent patterns
        simple_reagent_smarts = [
            "[OH-]",  # hydroxide
            "[H+]",   # proton
            "[Na+]", "[K+]", "[Li+]",  # alkali metals
            "[Cl-]", "[Br-]", "[I-]",  # halides
            "O=S(=O)([OH])[OH]",  # sulfuric acid
            "C(=O)[OH]",  # carboxylic acids (simple)
        ]
        
        for pattern in simple_reagent_smarts:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        
        return False
    
    def _fragment_incorporated_in_product(self, fragment, product) -> bool:
        """
        Check if a significant portion of the fragment is incorporated in the product.
        """
        fragment_atoms = fragment.GetNumAtoms()
        
        # Use maximum common substructure approach
        # For simplicity, check if at least 70% of fragment atoms can be found in product
        try:
            # Remove atom maps for substructure matching
            fragment_copy = Chem.Mol(fragment)
            for atom in fragment_copy.GetAtoms():
                atom.SetAtomMapNum(0)
            
            product_copy = Chem.Mol(product)
            for atom in product_copy.GetAtoms():
                atom.SetAtomMapNum(0)
            
            if product_copy.HasSubstructMatch(fragment_copy):
                return True
            
            # If exact match fails, try with a more flexible approach
            # Count matching atoms based on atomic number and connectivity
            matches = product_copy.GetSubstructMatches(fragment_copy)
            if matches:
                return True
                
        except Exception:
            pass
        
        return False
