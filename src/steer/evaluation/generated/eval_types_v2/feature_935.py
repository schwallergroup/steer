"""Generated evaluation code for: Benzoate protecting group strategy for sugar hydroxyls"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzoateProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of benzoate protecting groups on sugar hydroxyls.
    
    This scorer identifies reactions where benzoate esters are formed or cleaved on sugar molecules,
    favoring routes that use this protecting group strategy at appropriate depths.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Benzoate protection strategy not used
        else:
            if self.condition_type == "bool":
                return 1  # Strategy found
            else:
                # Early-stage protection is generally better
                return max(0, 1 - x)
    
    def hit_condition(self, d):
        """
        Detects benzoate ester formation/cleavage reactions on sugar substrates.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse molecules
            prod_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".") if smi]
            react_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".") if smi]
            
            if not all(prod_mols) or not all(react_mols):
                return False
                
            # Check if any molecule contains sugar-like structure
            sugar_pattern = Chem.MolFromSmarts("[CH1,CH2]-[CH1](-[OH,O])-[CH1](-[OH,O])-[CH1,CH2]")  # Basic sugar backbone
            has_sugar = any(mol.HasSubstructMatch(sugar_pattern) for mol in prod_mols + react_mols)
            
            if not has_sugar:
                return False
                
            # Benzoate ester pattern - COC(=O)c1ccccc1
            benzoate_pattern = Chem.MolFromSmarts("COC(=O)c1ccccc1")
            
            # Count benzoate groups in products vs reactants
            prod_benzoate_count = sum(len(mol.GetSubstructMatches(benzoate_pattern)) 
                                    for mol in prod_mols if mol.HasSubstructMatch(benzoate_pattern))
            react_benzoate_count = sum(len(mol.GetSubstructMatches(benzoate_pattern)) 
                                     for mol in react_mols if mol.HasSubstructMatch(benzoate_pattern))
            
            # Benzoate formation (protection) or cleavage (deprotection)
            if prod_benzoate_count > react_benzoate_count:
                return True  # Benzoate ester formation
            elif prod_benzoate_count < react_benzoate_count:
                return True  # Benzoate ester cleavage
                
            # Additional check for benzoyl chloride/benzoic acid reagents (common benzoate sources)
            benzoyl_chloride = Chem.MolFromSmarts("ClC(=O)c1ccccc1")
            benzoic_acid = Chem.MolFromSmarts("OC(=O)c1ccccc1")
            
            has_benzoyl_reagent = any(mol.HasSubstructMatch(benzoyl_chloride) or 
                                    mol.HasSubstructMatch(benzoic_acid) 
                                    for mol in react_mols)
            
            if has_benzoyl_reagent and prod_benzoate_count > 0:
                return True
                
            return False
            
        except Exception:
            return False
