"""Generated evaluation code for: Chemioselective ester hydrolysis final step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChemioselectiveEsterHydrolysis(BaseScoring):
    """
    Evaluates whether a chemioselective ester hydrolysis occurs as the final step.
    Specifically checks for selective hydrolysis of benzoate versus acetate esters.
    """
    
    def __init__(self, config: Dict):
        self.stage = config["parameters"].get("stage", "final")
        self.selectivity_type = config["parameters"].get("selectivity_type", "chemoselective")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        elif self.stage == "final":
            # For final step, we want x to be close to 0 (earliest/final step)
            return 10 * (1 - x) if x <= 1 else 0
        else:
            # For non-final, prefer earlier occurrence
            return 10 * (1 - min(x, 1))
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves chemioselective ester hydrolysis"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants, products = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mol = Chem.MolFromSmiles(reactants)
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
            
            if not reactant_mol or not product_mols:
                return False
                
            # Check for ester hydrolysis pattern
            if not self._is_ester_hydrolysis(reactant_mol, product_mols):
                return False
                
            # Check for chemioselectivity (presence of both benzoate and acetate patterns)
            return self._has_chemioselectivity(reactant_mol, product_mols)
            
        except:
            return False
    
    def _is_ester_hydrolysis(self, reactant, products):
        """Check if reaction involves ester hydrolysis (ester -> acid/alcohol + base products)"""
        # Ester pattern: R-COO-R'
        ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C,c]")
        
        # Carboxylic acid pattern: R-COOH or carboxylate R-COO-
        acid_pattern = Chem.MolFromSmarts("[C](=[O])[O]")
        
        # Check reactant has ester
        if not reactant.HasSubstructMatch(ester_pattern):
            return False
            
        # Check products contain carboxylic acid/carboxylate
        has_acid = any(mol.HasSubstructMatch(acid_pattern) for mol in products)
        
        # Look for alcohol product or hydroxide
        alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
        hydroxide_pattern = Chem.MolFromSmarts("[OH-]")
        
        has_alcohol_or_base = any(
            mol.HasSubstructMatch(alcohol_pattern) or mol.HasSubstructMatch(hydroxide_pattern) 
            for mol in products
        )
        
        return has_acid and has_alcohol_or_base
    
    def _has_chemioselectivity(self, reactant, products):
        """Check for chemioselective hydrolysis between benzoate and acetate esters"""
        # Benzoate ester pattern: Ar-COO-R
        benzoate_pattern = Chem.MolFromSmarts("[c]([C](=[O])[O][C,c])")
        
        # Acetate ester pattern: CH3-COO-R  
        acetate_pattern = Chem.MolFromSmarts("[CH3][C](=[O])[O][C,c]")
        
        # Check if reactant has both types of esters
        has_benzoate = reactant.HasSubstructMatch(benzoate_pattern)
        has_acetate = reactant.HasSubstructMatch(acetate_pattern)
        
        if not (has_benzoate and has_acetate):
            return False
            
        # Check selectivity by examining which ester was hydrolyzed
        # Look for benzoic acid or benzoate in products (indicating benzoate hydrolysis)
        benzoic_acid_pattern = Chem.MolFromSmarts("[c]([C](=[O])[OH,O-])")
        
        # Look for acetic acid or acetate in products (indicating acetate hydrolysis)  
        acetic_acid_pattern = Chem.MolFromSmarts("[CH3][C](=[O])[OH,O-]")
        
        has_benzoic_product = any(mol.HasSubstructMatch(benzoic_acid_pattern) for mol in products)
        has_acetic_product = any(mol.HasSubstructMatch(acetic_acid_pattern) for mol in products)
        
        # Chemioselective if only one type was hydrolyzed
        return has_benzoic_product ^ has_acetic_product
