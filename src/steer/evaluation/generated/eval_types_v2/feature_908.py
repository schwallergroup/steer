"""Generated evaluation code for: Benzyl protecting group with halides present"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectionWithHalides(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the presence of benzyl protecting groups 
    when halides are present, which creates a problematic combination since
    hydrogenolytic deprotection would also remove the halides.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.allow_combination = config.get("allow_combination", False)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check if route contains both benzyl protection and halides
        has_benzyl_protection = any(self.detect_benzyl_protection(r) for r in reactions)
        has_halides = any(self.detect_halides_present(r) for r in reactions)
        
        # The problematic combination exists if both are present
        problematic_combination = has_benzyl_protection and has_halides
        
        # Condition is met based on whether we allow this combination
        condition = problematic_combination != self.allow_combination
        
        return condition, len(reactions)
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl ether or ester protection formation or presence"""
        # Benzyl ether pattern (Ar-CH2-O-R)
        benzyl_ether_pattern = "c1ccccc1CO"
        # Benzyl ester pattern (Ar-CH2-O-C(=O))
        benzyl_ester_pattern = "c1ccccc1COC(=O)"
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if benzyl protection is formed (appears in products but not reactants)
        # or if benzyl-protected species are present
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Remove None molecules (parsing failures)
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            benzyl_ether_smarts = Chem.MolFromSmarts(benzyl_ether_pattern)
            benzyl_ester_smarts = Chem.MolFromSmarts(benzyl_ester_pattern)
            
            # Check for benzyl protection in products
            for mol in product_mols:
                if mol.HasSubstructMatch(benzyl_ether_smarts) or mol.HasSubstructMatch(benzyl_ester_smarts):
                    return True
                    
            return False
            
        except:
            return False
    
    def detect_halides_present(self, rxn):
        """Detect presence of halides (F, Cl, Br, I) in the reaction"""
        # Halide patterns - aromatic and aliphatic
        halide_patterns = [
            "[F,Cl,Br,I]",  # Any halide
            "c[F,Cl,Br,I]", # Aromatic halide
            "C[F,Cl,Br,I]"  # Aliphatic halide
        ]
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        # Check both reactants and products for halides
        all_smiles = rxn_parts[0] + "." + rxn_parts[1]
        
        try:
            mols = [Chem.MolFromSmiles(smi.strip()) for smi in all_smiles.split(".")]
            mols = [mol for mol in mols if mol is not None]
            
            for pattern in halide_patterns:
                halide_smarts = Chem.MolFromSmarts(pattern)
                for mol in mols:
                    if mol.HasSubstructMatch(halide_smarts):
                        return True
                        
            return False
            
        except:
            return False
