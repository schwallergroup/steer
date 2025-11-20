"""Generated evaluation code for: Orthogonal ester protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalEsterProtection(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route employs orthogonal ester protecting group strategy.
    Checks for the presence of both tert-butyl and ethyl esters with their respective
    deprotection reactions under different conditions (acidic vs basic).
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["tert-butyl ester", "ethyl ester"])
        self.deprotection_conditions = config.get("deprotection_conditions", ["acidic", "basic"])
        self.selectivity = config.get("selectivity", "orthogonal")
        
        # SMARTS patterns for ester protecting groups
        self.tert_butyl_ester_pattern = "C(=O)OC(C)(C)C"
        self.ethyl_ester_pattern = "C(=O)OCC"
        
        # SMARTS patterns for deprotection products (carboxylic acids)
        self.carboxylic_acid_pattern = "C(=O)O"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_tert_butyl_protection = False
        has_ethyl_protection = False
        has_tert_butyl_deprotection = False
        has_ethyl_deprotection = False
        has_orthogonal_conditions = False
        
        # Track deprotection conditions for orthogonality check
        tert_butyl_deprotection_acidic = False
        ethyl_deprotection_basic = False
        
        for rxn in reactions:
            # Check for protection reactions (formation of esters)
            if self.detect_ester_formation(rxn, self.tert_butyl_ester_pattern):
                has_tert_butyl_protection = True
            elif self.detect_ester_formation(rxn, self.ethyl_ester_pattern):
                has_ethyl_protection = True
                
            # Check for deprotection reactions (ester hydrolysis)
            if self.detect_ester_deprotection(rxn, self.tert_butyl_ester_pattern):
                has_tert_butyl_deprotection = True
                # tert-butyl esters typically cleaved under acidic conditions
                if self.is_acidic_deprotection(rxn):
                    tert_butyl_deprotection_acidic = True
                    
            elif self.detect_ester_deprotection(rxn, self.ethyl_ester_pattern):
                has_ethyl_deprotection = True
                # ethyl esters typically cleaved under basic conditions
                if self.is_basic_deprotection(rxn):
                    ethyl_deprotection_basic = True
        
        # Check for orthogonal selectivity
        if self.selectivity == "orthogonal":
            has_orthogonal_conditions = tert_butyl_deprotection_acidic and ethyl_deprotection_basic
        
        # Condition is met if both protecting groups are used with orthogonal deprotection
        condition = (has_tert_butyl_protection and has_ethyl_protection and 
                    has_tert_butyl_deprotection and has_ethyl_deprotection and
                    has_orthogonal_conditions)
        
        return condition, len(reactions)
    
    def detect_ester_formation(self, rxn, ester_pattern):
        """Detect formation of specific ester protecting group."""
        try:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if ester pattern appears in products but not in reactants
            ester_mol = Chem.MolFromSmarts(ester_pattern)
            if ester_mol is None:
                return False
                
            has_ester_in_products = any(
                Chem.MolFromSmiles(p) and Chem.MolFromSmiles(p).HasSubstructMatch(ester_mol)
                for p in products if p.strip()
            )
            
            has_ester_in_reactants = any(
                Chem.MolFromSmiles(r) and Chem.MolFromSmiles(r).HasSubstructMatch(ester_mol)
                for r in reactants if r.strip()
            )
            
            return has_ester_in_products and not has_ester_in_reactants
            
        except Exception:
            return False
    
    def detect_ester_deprotection(self, rxn, ester_pattern):
        """Detect deprotection (cleavage) of specific ester protecting group."""
        try:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            ester_mol = Chem.MolFromSmarts(ester_pattern)
            acid_mol = Chem.MolFromSmarts(self.carboxylic_acid_pattern)
            
            if ester_mol is None or acid_mol is None:
                return False
            
            # Check if ester is consumed and carboxylic acid is formed
            has_ester_in_reactants = any(
                Chem.MolFromSmiles(r) and Chem.MolFromSmiles(r).HasSubstructMatch(ester_mol)
                for r in reactants if r.strip()
            )
            
            has_acid_in_products = any(
                Chem.MolFromSmiles(p) and Chem.MolFromSmiles(p).HasSubstructMatch(acid_mol)
                for p in products if p.strip()
            )
            
            return has_ester_in_reactants and has_acid_in_products
            
        except Exception:
            return False
    
    def is_acidic_deprotection(self, rxn):
        """Check if reaction involves acidic conditions (heuristic based on reagents)."""
        try:
            # Common acidic reagents for tert-butyl ester deprotection
            acidic_reagents = ["TFA", "HCl", "H2SO4", "CF3COOH", "trifluoroacetic acid"]
            rxn_lower = rxn.lower()
            return any(reagent.lower() in rxn_lower for reagent in acidic_reagents)
        except Exception:
            return False
    
    def is_basic_deprotection(self, rxn):
        """Check if reaction involves basic conditions (heuristic based on reagents)."""
        try:
            # Common basic reagents for ethyl ester deprotection
            basic_reagents = ["NaOH", "KOH", "LiOH", "Ba(OH)2", "sodium hydroxide", "potassium hydroxide"]
            rxn_lower = rxn.lower()
            return any(reagent.lower() in rxn_lower for reagent in basic_reagents)
        except Exception:
            return False
