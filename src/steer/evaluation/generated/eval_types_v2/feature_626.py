"""Generated evaluation code for: Late stage SNAr ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylEtherSNAr(BaseScoring):
    """
    Evaluates synthesis routes for late-stage aryl ether formation via nucleophilic aromatic substitution.
    Detects SNAr reactions that form C-O bonds between aromatic carbons and oxygen nucleophiles.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.9)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        """
        Score based on how late in the synthesis the SNAr ether formation occurs.
        Late-stage (high depth fraction) is preferred.
        """
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.condition_type == "bool":
            return 1 if x >= 0.7 else 0  # Consider late if in final 30% of route
        else:
            # Reward reactions closer to the target depth (late stage)
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents SNAr ether formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
                
            # Check for SNAr ether formation
            return self._is_snar_ether_formation(reactants, products)
            
        except Exception:
            return False
    
    def _is_snar_ether_formation(self, reactants, products) -> bool:
        """
        Detect if this is an SNAr reaction forming an aryl ether bond.
        """
        # Pattern for aryl halide (F, Cl, Br, I on aromatic carbon)
        aryl_halide_pattern = Chem.MolFromSmarts("[cH0,c:1]-[F,Cl,Br,I]")
        
        # Pattern for oxygen nucleophile (alcohol, phenol, alkoxide)
        oxygen_nucleophile_pattern = Chem.MolFromSmarts("[OH1,O-:2]")
        
        # Pattern for aryl ether product
        aryl_ether_pattern = Chem.MolFromSmarts("[c:1]-[O:2]")
        
        if not all([aryl_halide_pattern, oxygen_nucleophile_pattern, aryl_ether_pattern]):
            return False
        
        # Check reactants for aryl halide and oxygen nucleophile
        has_aryl_halide = False
        has_oxygen_nucleophile = False
        aryl_carbon_map = None
        oxygen_map = None
        
        for reactant in reactants:
            # Check for aryl halide
            if reactant.HasSubstructMatch(aryl_halide_pattern):
                has_aryl_halide = True
                match = reactant.GetSubstructMatch(aryl_halide_pattern)
                if match:
                    aryl_atom = reactant.GetAtomWithIdx(match[0])
                    aryl_carbon_map = aryl_atom.GetAtomMapNum()
            
            # Check for oxygen nucleophile
            if reactant.HasSubstructMatch(oxygen_nucleophile_pattern):
                has_oxygen_nucleophile = True
                match = reactant.GetSubstructMatch(oxygen_nucleophile_pattern)
                if match:
                    oxygen_atom = reactant.GetAtomWithIdx(match[0])
                    oxygen_map = oxygen_atom.GetAtomMapNum()
        
        if not (has_aryl_halide and has_oxygen_nucleophile):
            return False
        
        # Check products for aryl ether formation with same atom maps
        for product in products:
            if product.HasSubstructMatch(aryl_ether_pattern):
                matches = product.GetSubstructMatches(aryl_ether_pattern)
                for match in matches:
                    carbon_atom = product.GetAtomWithIdx(match[0])
                    oxygen_atom = product.GetAtomWithIdx(match[1])
                    
                    # Verify atom mapping consistency
                    if (carbon_atom.GetAtomMapNum() == aryl_carbon_map and 
                        oxygen_atom.GetAtomMapNum() == oxygen_map):
                        return True
        
        return False
