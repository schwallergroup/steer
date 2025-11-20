"""Generated evaluation code for: Convergent synthesis via amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCoupling(BaseScoring):
    """
    Evaluates convergent synthesis strategy via amide coupling at a specific step.
    Checks if amide bond formation occurs at the target step with appropriate fragment complexity.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["step_number"]
        self.fragment_complexity = config["parameters"]["fragment_complexity"]
        self.min_complexity_threshold = 10 if self.fragment_complexity == "high" else 5
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't occur
        else:
            # Reward when coupling occurs at target step
            step_penalty = abs(x - (self.target_step / 10.0))  # Normalize to 0-1 range
            return max(0, 1 - step_penalty) * 10  # Scale to 0-10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is an amide coupling with appropriate complexity"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        # Check if amide bond is formed
        if not self._is_amide_coupling(mapped_rxn):
            return False
            
        # Check fragment complexity for convergent synthesis
        return self._has_convergent_complexity(mapped_rxn)
    
    def _is_amide_coupling(self, mapped_rxn: str) -> bool:
        """Detect amide bond formation in the reaction"""
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product = Chem.MolFromSmiles(rxn_parts[0])
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
        
        if not product or not all(reactants):
            return False
        
        # Count amide bonds in product vs reactants
        amide_pattern = Chem.MolFromSmarts("[C](=O)[NH]")
        product_amides = len(product.GetSubstructMatches(amide_pattern))
        reactant_amides = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
        
        # Amide coupling should increase amide bond count
        return product_amides > reactant_amides
    
    def _has_convergent_complexity(self, mapped_rxn: str) -> bool:
        """Check if both fragments meet complexity threshold for convergent synthesis"""
        rxn_parts = mapped_rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".") if r]
        
        if len(reactants) < 2:
            return False
        
        # Filter out small molecules (catalysts, coupling reagents)
        major_fragments = [r for r in reactants if r.GetNumAtoms() >= 5]
        
        if len(major_fragments) < 2:
            return False
        
        # Check complexity of two largest fragments
        complexities = [self._calculate_complexity(frag) for frag in major_fragments]
        complexities.sort(reverse=True)
        
        # Both major fragments should meet minimum complexity
        return (len(complexities) >= 2 and 
                complexities[0] >= self.min_complexity_threshold and 
                complexities[1] >= self.min_complexity_threshold)
    
    def _calculate_complexity(self, mol) -> int:
        """Simple complexity metric based on rings, heteroatoms, and size"""
        if not mol:
            return 0
        
        complexity = 0
        complexity += mol.GetNumAtoms()  # Size
        complexity += Chem.rdMolDescriptors.CalcNumRings(mol) * 3  # Ring systems
        complexity += sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != 'C') * 2  # Heteroatoms
        complexity += len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))  # Stereocenters
        
        return complexity
