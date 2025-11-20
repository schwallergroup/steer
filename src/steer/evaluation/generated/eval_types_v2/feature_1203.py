"""Generated evaluation code for: Late piperazine ring formation via intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperazineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage piperazine ring formation via intramolecular cyclization.
    Detects formation of piperazine rings (C1CNCCN1) through intramolecular double SN2 cyclization
    of bis(2-chloroethyl)amine precursors.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CNCCN1")
        self.timing = config.get("timing", "late")
        self.formation_method = config.get("formation_method", "intramolecular_cyclization")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Piperazine formation doesn't occur
        
        if self.timing == "late":
            # Late formation is preferred, so higher depth fraction is better
            return 10 * x  # Scale 0-1 to 0-10, favoring later stages
        else:
            # Early formation preferred
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a piperazine ring via intramolecular cyclization.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains piperazine ring
            piperazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not product_mol.HasSubstructMatch(piperazine_pattern):
                return False
            
            # Check if this is intramolecular cyclization by verifying:
            # 1. Single reactant molecule (intramolecular)
            # 2. Contains bis(2-chloroethyl)amine or similar cyclizable precursor
            reactants = [r.strip() for r in reactant_smiles.split(".")]
            
            # Filter out small molecules (catalysts, bases, etc.)
            main_reactants = []
            for r_smiles in reactants:
                r_mol = Chem.MolFromSmiles(r_smiles)
                if r_mol and r_mol.GetNumAtoms() > 5:  # Ignore small molecules
                    main_reactants.append(r_mol)
            
            # Should be primarily intramolecular (one main reactant)
            if len(main_reactants) != 1:
                return False
                
            main_reactant = main_reactants[0]
            
            # Check for cyclizable precursor patterns
            # bis(2-chloroethyl)amine pattern: N(CCCl)CCCl
            precursor_patterns = [
                "N(CCCl)CCCl",  # bis(2-chloroethyl)amine
                "N(CCBr)CCBr",  # bis(2-bromoethyl)amine
                "N(CC[OH])CC[OH]",  # bis(2-hydroxyethyl)amine (with activating conditions)
                "N(CCO)CCO",    # alternative hydroxyl pattern
            ]
            
            for pattern_smarts in precursor_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and main_reactant.HasSubstructMatch(pattern):
                    return True
                    
            return False
            
        except Exception:
            return False
