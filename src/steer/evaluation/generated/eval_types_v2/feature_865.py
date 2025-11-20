"""Generated evaluation code for: Sequential stereoinversion via Mitsunobu-saponification"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MitsunobuSaponificationSequence(MultiRxnCondBase):
    """
    Detects sequential stereoinversion via Mitsunobu-saponification reaction sequence.
    Looks for Mitsunobu esterification followed by saponification on secondary alcohols
    for stereochemical inversion purposes.
    """
    
    def __init__(self, config):
        self.require_sequence = config.get("require_sequence", True)
        self.max_steps_between = config.get("max_steps_between", 3)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        mitsunobu_steps = []
        saponification_steps = []
        
        # Find all Mitsunobu and saponification reactions
        for i, rxn in enumerate(reactions):
            if self.detect_mitsunobu(rxn):
                mitsunobu_steps.append(i)
            if self.detect_saponification(rxn):
                saponification_steps.append(i)
        
        # Check if we have the required sequence
        has_sequence = False
        if self.require_sequence:
            has_sequence = self.has_mitsunobu_saponification_sequence(
                mitsunobu_steps, saponification_steps
            )
        else:
            has_sequence = len(mitsunobu_steps) > 0 and len(saponification_steps) > 0
        
        return has_sequence, len(reactions)
    
    def has_mitsunobu_saponification_sequence(self, mitsunobu_steps, saponification_steps):
        """Check if Mitsunobu is followed by saponification within max_steps_between"""
        for m_step in mitsunobu_steps:
            for s_step in saponification_steps:
                if s_step > m_step and (s_step - m_step) <= self.max_steps_between:
                    return True
        return False
    
    def detect_mitsunobu(self, rxn):
        """Detect Mitsunobu reaction pattern"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Look for DEAD/DIAD (diethyl/diisopropyl azodicarboxylate) patterns
            dead_pattern = "CCOC(=O)N=NC(=O)OCC"  # DEAD
            diad_pattern = "CC(C)OC(=O)N=NC(=O)OC(C)C"  # DIAD
            
            # Look for PPh3 (triphenylphosphine)
            pph3_pattern = "c1ccc(P(c2ccccc2)c2ccccc2)cc1"
            
            # Look for secondary alcohol -> ester conversion with inversion
            sec_alcohol_pattern = "[CH]([CH3,CH2,c])O"
            
            has_azo = dead_pattern in reactants or diad_pattern in reactants
            has_pph3 = pph3_pattern in reactants
            
            # Check for alcohol to ester conversion
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Look for secondary alcohol in reactants
            sec_alcohol_smarts = Chem.MolFromSmarts("[CH]([*])O")
            has_sec_alcohol = any(mol.HasSubstructMatch(sec_alcohol_smarts) for mol in reactant_mols)
            
            # Look for ester in products
            ester_smarts = Chem.MolFromSmarts("C(=O)O[CH]([*])")
            has_ester = any(mol.HasSubstructMatch(ester_smarts) for mol in product_mols)
            
            return (has_azo or has_pph3) and has_sec_alcohol and has_ester
            
        except Exception:
            return False
    
    def detect_saponification(self, rxn):
        """Detect saponification reaction pattern"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Look for base (OH-, NaOH, KOH, LiOH)
            base_patterns = ["[OH-]", "[Na+]", "[K+]", "[Li+]", "NaOH", "KOH", "LiOH"]
            has_base = any(pattern in reactants for pattern in base_patterns)
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Look for ester in reactants
            ester_smarts = Chem.MolFromSmarts("C(=O)O[CH]([*])")
            has_ester_reactant = any(mol.HasSubstructMatch(ester_smarts) for mol in reactant_mols)
            
            # Look for alcohol in products (stereoinversion product)
            alcohol_smarts = Chem.MolFromSmarts("[CH]([*])O")
            has_alcohol_product = any(mol.HasSubstructMatch(alcohol_smarts) for mol in product_mols)
            
            # Look for carboxylate/carboxylic acid in products
            carboxylate_smarts = Chem.MolFromSmarts("C(=O)[O-,OH]")
            has_carboxylate = any(mol.HasSubstructMatch(carboxylate_smarts) for mol in product_mols)
            
            return has_base and has_ester_reactant and has_alcohol_product and has_carboxylate
            
        except Exception:
            return False
