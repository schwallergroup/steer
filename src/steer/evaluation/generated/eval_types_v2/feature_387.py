"""Generated evaluation code for: Selective mono-alkylation of dibromide substrate"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SelectiveMonoAlkylationDibromide(BaseScoring):
    """
    Evaluates selective mono-alkylation of dibromide substrates via Williamson ether synthesis.
    Checks if a dibromide substrate undergoes selective ether formation at one position
    while leaving the other bromide intact.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        self.substrate_pattern = config.get("substrate_pattern", "*CBr.*CBr")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        if self.condition_type == "bool":
            return 1  # Condition met
        else:
            # Earlier selective alkylation is better (closer to 0)
            return max(0, 1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents selective mono-alkylation of dibromide"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactant_smiles = mapped_rxn.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactants):
                return False
            
            # Check if product has dibromide pattern
            dibromide_pattern = Chem.MolFromSmarts(self.substrate_pattern)
            if not dibromide_pattern or not product_mol.HasSubstructMatch(dibromide_pattern):
                return False
            
            # Check if any reactant has mono-bromide (one Br converted to ether)
            monobromide_ether_found = False
            
            for reactant in reactants:
                # Count bromines in reactant vs product
                reactant_br_count = len([atom for atom in reactant.GetAtoms() if atom.GetSymbol() == "Br"])
                product_br_count = len([atom for atom in product_mol.GetAtoms() if atom.GetSymbol() == "Br"])
                
                # Check for ether formation (C-O-C pattern increase)
                ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
                reactant_ether_count = len(reactant.GetSubstructMatches(ether_pattern))
                product_ether_count = len(product_mol.GetSubstructMatches(ether_pattern))
                
                # Selective mono-alkylation: one less Br, one more ether
                if (product_br_count == reactant_br_count - 1 and 
                    product_ether_count > reactant_ether_count and
                    reactant_br_count >= 2):
                    
                    # Verify still has at least one Br remaining
                    if product_br_count >= 1:
                        monobromide_ether_found = True
                        break
            
            return monobromide_ether_found
            
        except Exception:
            return False
