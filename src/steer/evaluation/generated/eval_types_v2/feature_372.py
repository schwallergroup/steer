"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two major fragments 
    are coupled via a specific reaction type to form a target bond.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "SNAr")
        self.bond_formed = config.get("bond_formed", "C-O")
        
        # Define reaction patterns for different coupling types
        self.reaction_patterns = {
            "SNAr": {
                "electrophile": "[#6]1:[#6]:[#6]([F,Cl,Br,I]):[#6]:[#6]:[#6]:1",  # Aryl halide
                "nucleophile": "[OH,SH,NH2,NH]",  # Nucleophile
                "product_bond": "[#6]-[#8,#16,#7]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1"  # C-heteroatom-aryl
            },
            "Suzuki": {
                "electrophile": "[#6]1:[#6]:[#6]([Br,I]):[#6]:[#6]:[#6]:1",
                "nucleophile": "[#6]-[B]([OH])([OH])",
                "product_bond": "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6]"
            },
            "Buchwald": {
                "electrophile": "[#6]1:[#6]:[#6]([Br,Cl,I]):[#6]:[#6]:[#6]:1",
                "nucleophile": "[NH2,NH]",
                "product_bond": "[#6]1:[#6]:[#6]([NH]):[#6]:[#6]:[#6]:1"
            }
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Earlier convergent coupling is better (more convergent)
            # Scale to 0-10 range, with early coupling (low x) getting high scores
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of major fragments
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
            
            # Check if we have the expected number of major fragments
            major_fragments = [r for r in reactants if r.GetNumAtoms() > 5]  # Filter small reagents
            if len(major_fragments) != self.fragment_count:
                return False
            
            # Check if this is the specified coupling reaction type
            if self.coupling_reaction in self.reaction_patterns:
                return self._check_coupling_reaction(reactants, product)
            
            return False
            
        except Exception:
            return False
    
    def _check_coupling_reaction(self, reactants, product) -> bool:
        """
        Check if reactants undergo the specified coupling reaction
        """
        patterns = self.reaction_patterns[self.coupling_reaction]
        
        # Look for electrophile and nucleophile patterns in reactants
        electrophile_found = False
        nucleophile_found = False
        
        electrophile_pattern = Chem.MolFromSmarts(patterns["electrophile"])
        nucleophile_pattern = Chem.MolFromSmarts(patterns["nucleophile"])
        product_pattern = Chem.MolFromSmarts(patterns["product_bond"])
        
        if not all([electrophile_pattern, nucleophile_pattern, product_pattern]):
            return False
        
        # Check reactants for coupling partners
        for reactant in reactants:
            if reactant.HasSubstructMatch(electrophile_pattern):
                electrophile_found = True
            if reactant.HasSubstructMatch(nucleophile_pattern):
                nucleophile_found = True
        
        # Check if product has the expected bond formation
        product_bond_formed = product.HasSubstructMatch(product_pattern)
        
        # Verify that we have both coupling partners and the expected product
        return electrophile_found and nucleophile_found and product_bond_formed
