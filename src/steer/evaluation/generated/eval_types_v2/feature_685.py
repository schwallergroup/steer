"""Generated evaluation code for: Ozonolysis with sulfide oxidation side reaction risk"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OzonolysisWithSulfideRisk(BaseScoring):
    """
    Evaluates routes containing ozonolysis reactions on substrates with thioether groups.
    Ozonolysis can cause unwanted sulfide oxidation as a side reaction when performed
    on molecules containing sulfur atoms.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 10  # No problematic ozonolysis found - best score
        else:
            # Penalize based on how late in the route this risky reaction occurs
            # Later occurrence is worse as more work could be lost
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents ozonolysis on a thioether-containing substrate
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            # Parse the product (starting material in retrosynthetic direction)
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains thioether pattern
            thioether_pattern = Chem.MolFromSmarts("[#6]-[#16]-[#6]")  # C-S-C thioether
            if not product_mol.HasSubstructMatch(thioether_pattern):
                return False
                
            # Check for ozonolysis pattern in the reaction
            # Ozonolysis typically breaks C=C double bonds, forming carbonyls
            reactants = reactant_smiles.split(".")
            
            # Look for ozonolysis indicators:
            # 1. Product has C=C that becomes C=O in reactants
            # 2. Or check for typical ozonolysis transformation patterns
            alkene_pattern = Chem.MolFromSmarts("C=C")
            carbonyl_pattern = Chem.MolFromSmarts("C=O")
            
            product_has_alkene = product_mol.HasSubstructMatch(alkene_pattern)
            
            # Check if reactants have more carbonyls than product (indicating ozonolysis)
            product_carbonyls = len(product_mol.GetSubstructMatches(carbonyl_pattern))
            total_reactant_carbonyls = 0
            
            for reactant_smi in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smi)
                if reactant_mol:
                    total_reactant_carbonyls += len(reactant_mol.GetSubstructMatches(carbonyl_pattern))
            
            # Ozonolysis likely if: product has alkene, reactants have more carbonyls
            is_ozonolysis = product_has_alkene and (total_reactant_carbonyls > product_carbonyls)
            
            return is_ozonolysis
            
        except Exception:
            return False
