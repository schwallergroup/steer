"""Generated evaluation code for: Early acyl chloride formation with unprotected nucleophiles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAcylChlorideWithNucleophiles(BaseScoring):
    """
    Evaluates synthesis routes for early acyl chloride formation in the presence of 
    unprotected nucleophiles (primary amines and secondary alcohols).
    
    Scores routes based on when acyl chloride formation occurs relative to the presence
    of competing nucleophilic functional groups.
    """
    
    def __init__(self, config: Dict):
        self.functional_groups = config["parameters"]["functional_groups_present"]
        self.protection_status = config["parameters"]["protection_status"]
        
        # SMARTS patterns for functional groups
        self.patterns = {
            "primary_amine": "[NH2][CX4]",  # Primary amine attached to sp3 carbon
            "secondary_alcohol": "[OH1][CX4H1]",  # OH on secondary carbon
            "acyl_chloride": "[CX3](=[OX1])[Cl]"  # Acyl chloride pattern
        }
    
    def route_scoring(self, x) -> float:
        """
        Score based on depth of acyl chloride formation.
        Early formation with unprotected nucleophiles is penalized.
        """
        if x < 0:
            return 0  # Condition not met
        
        # Earlier formation (higher x) gets lower score when nucleophiles present
        # Scale from 0-10, where early formation = low score
        return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step forms an acyl chloride while 
        unprotected nucleophiles are present.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) 
                          for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check if acyl chloride is formed (present in products but not reactants)
            acyl_chloride_pattern = Chem.MolFromSmarts(self.patterns["acyl_chloride"])
            
            has_acyl_chloride_product = any(mol.HasSubstructMatch(acyl_chloride_pattern) 
                                          for mol in product_mols)
            has_acyl_chloride_reactant = any(mol.HasSubstructMatch(acyl_chloride_pattern) 
                                           for mol in reactant_mols)
            
            # Acyl chloride formation occurs if it's in products but not in reactants
            acyl_chloride_formed = has_acyl_chloride_product and not has_acyl_chloride_reactant
            
            if not acyl_chloride_formed:
                return False
            
            # Check for presence of unprotected nucleophiles in the reaction mixture
            if self.protection_status == "unprotected":
                nucleophiles_present = False
                
                for fg in self.functional_groups:
                    if fg in self.patterns:
                        pattern = Chem.MolFromSmarts(self.patterns[fg])
                        
                        # Check both reactants and products for nucleophiles
                        all_mols = reactant_mols + product_mols
                        if any(mol.HasSubstructMatch(pattern) for mol in all_mols):
                            nucleophiles_present = True
                            break
                
                return nucleophiles_present
            
            return False
            
        except Exception:
            return False
