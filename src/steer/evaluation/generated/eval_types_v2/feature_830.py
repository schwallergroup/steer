"""Generated evaluation code for: Early stage electrophilic bromination on imidazole"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyElectrophilicBromination(BaseScoring):
    """
    Evaluates whether electrophilic bromination on imidazole occurs at early stage.
    Checks for bromination reactions where an imidazole substrate gains a bromine atom
    through electrophilic aromatic substitution.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config["parameters"]["substrate_pattern"]
        self.stage = config["parameters"]["stage"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        
        if self.stage == "early":
            return 1 - x  # Earlier is better, so invert the depth fraction
        elif self.stage == "late":
            return x  # Later is better
        else:
            return 1 if x >= 0 else 0  # Just presence/absence
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction is an electrophilic bromination on imidazole.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for imidazole substrate pattern in reactants
            imidazole_pattern = Chem.MolFromSmarts(self.substrate_pattern)
            if not imidazole_pattern:
                return False
                
            imidazole_reactant = None
            for reactant in reactants:
                if reactant.HasSubstructMatch(imidazole_pattern):
                    imidazole_reactant = reactant
                    break
                    
            if not imidazole_reactant:
                return False
            
            # Check if brominated product is formed
            brominated_product = None
            for product in products:
                if product.HasSubstructMatch(imidazole_pattern):
                    # Count bromine atoms in reactant vs product
                    reactant_br_count = sum(1 for atom in imidazole_reactant.GetAtoms() 
                                          if atom.GetSymbol() == 'Br')
                    product_br_count = sum(1 for atom in product.GetAtoms() 
                                         if atom.GetSymbol() == 'Br')
                    
                    if product_br_count > reactant_br_count:
                        brominated_product = product
                        break
            
            if not brominated_product:
                return False
                
            # Additional check: ensure this looks like electrophilic aromatic substitution
            # (bromine should be attached to carbon in the aromatic ring)
            bromine_on_aromatic = False
            for atom in brominated_product.GetAtoms():
                if atom.GetSymbol() == 'Br':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
                            bromine_on_aromatic = True
                            break
                    if bromine_on_aromatic:
                        break
                        
            return bromine_on_aromatic
            
        except Exception:
            return False
