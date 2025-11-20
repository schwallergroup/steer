"""Generated evaluation code for: Selective ester hydrolysis strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SelectiveEsterHydrolysis(BaseScoring):
    """
    Evaluates routes for selective ester hydrolysis strategy.
    Checks if aromatic esters are hydrolyzed while aliphatic esters are preserved,
    particularly focusing on proline-type aliphatic esters.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Selective hydrolysis doesn't occur
        else:
            # Earlier selective hydrolysis is generally better for synthetic strategy
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves selective ester hydrolysis"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactant_smiles, product_smiles = mapped_rxn.split(">>")
            reactant = Chem.MolFromSmiles(reactant_smiles)
            products = [Chem.MolFromSmiles(p) for p in product_smiles.split(".")]
            
            if not reactant or not all(products):
                return False
                
            # Check if this is an ester hydrolysis reaction
            if not self._is_ester_hydrolysis(reactant, products):
                return False
                
            # Check for selective hydrolysis pattern
            return self._is_selective_aromatic_vs_aliphatic(reactant, products)
            
        except Exception:
            return False
    
    def _is_ester_hydrolysis(self, reactant, products):
        """Check if reaction involves ester hydrolysis"""
        # Look for ester pattern in reactant
        ester_pattern = Chem.MolFromSmarts("[C,c](=[O])-[O]-[C,c]")
        if not reactant.HasSubstructMatch(ester_pattern):
            return False
            
        # Check if products contain carboxylic acid and alcohol/phenol
        acid_pattern = Chem.MolFromSmarts("[C,c](=[O])-[OH]")
        alcohol_pattern = Chem.MolFromSmarts("[C,c]-[OH]")
        
        has_acid = any(p.HasSubstructMatch(acid_pattern) for p in products)
        has_alcohol = any(p.HasSubstructMatch(alcohol_pattern) for p in products)
        
        return has_acid and has_alcohol
    
    def _is_selective_aromatic_vs_aliphatic(self, reactant, products):
        """Check if aromatic ester is hydrolyzed while aliphatic ester is preserved"""
        # Aromatic ester patterns
        aromatic_ester = Chem.MolFromSmarts("c(=[O])-[O]-[C,c]")  # Aromatic carbonyl
        phenyl_ester = Chem.MolFromSmarts("[C,c](=[O])-[O]-c")    # Ester to aromatic
        
        # Aliphatic ester patterns (especially proline-like)
        aliphatic_ester = Chem.MolFromSmarts("[CH2,CH3]-[O]-C(=[O])-C")
        proline_ester = Chem.MolFromSmarts("[CH2,CH3]-[O]-C(=[O])-[CH]1-[CH2,CH]-[CH2,CH]-[CH2,CH]-N1")
        
        # Count esters in reactant
        aromatic_esters_reactant = (reactant.GetSubstructMatches(aromatic_ester) + 
                                  reactant.GetSubstructMatches(phenyl_ester))
        aliphatic_esters_reactant = (reactant.GetSubstructMatches(aliphatic_ester) + 
                                   reactant.GetSubstructMatches(proline_ester))
        
        # Must have both types in reactant
        if not (aromatic_esters_reactant and aliphatic_esters_reactant):
            return False
            
        # Count esters remaining in products
        all_products = Chem.MolFromSmiles(".".join([Chem.MolToSmiles(p) for p in products]))
        if not all_products:
            return False
            
        aromatic_esters_products = (all_products.GetSubstructMatches(aromatic_ester) + 
                                  all_products.GetSubstructMatches(phenyl_ester))
        aliphatic_esters_products = (all_products.GetSubstructMatches(aliphatic_ester) + 
                                   all_products.GetSubstructMatches(proline_ester))
        
        # Selective hydrolysis: aromatic esters reduced, aliphatic esters preserved
        aromatic_hydrolyzed = len(aromatic_esters_reactant) > len(aromatic_esters_products)
        aliphatic_preserved = len(aliphatic_esters_reactant) == len(aliphatic_esters_products)
        
        return aromatic_hydrolyzed and aliphatic_preserved
