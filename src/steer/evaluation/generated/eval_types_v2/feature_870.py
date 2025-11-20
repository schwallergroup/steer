"""Generated evaluation code for: Alcohol protection before N-arylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholProtectionNArylation(BaseScoring):
    """
    Evaluates if alcohol acetate protection occurs before Chan-Lam N-arylation coupling.
    Checks for the presence of acetate protection followed by Chan-Lam coupling in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "relative")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
        self.acetate_pattern = Chem.MolFromSmarts("[OH1][C](=O)[CH3]")  # Acetate ester pattern
        self.alcohol_pattern = Chem.MolFromSmarts("[OH1][C]")  # Simple alcohol pattern
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is better (lower depth is higher score)
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d):
        """Check if this reaction involves alcohol acetate protection before Chan-Lam coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not react_mols:
                return False
                
            # Check if this is an acetate protection reaction
            if self._is_acetate_protection(prod_mol, react_mols):
                # Check if Chan-Lam coupling occurs later in the route
                return self._has_subsequent_chan_lam(d)
                
        except Exception:
            return False
            
        return False
    
    def _is_acetate_protection(self, product, reactants):
        """Check if reaction converts alcohol to acetate ester"""
        # Product should have acetate group
        if not product.HasSubstructMatch(self.acetate_pattern):
            return False
            
        # At least one reactant should have free alcohol
        has_alcohol_reactant = any(mol.HasSubstructMatch(self.alcohol_pattern) for mol in reactants)
        
        # Check for acetylating reagent (acetic anhydride or acetyl chloride)
        acetyl_anhydride = Chem.MolFromSmarts("CC(=O)OC(=O)C")
        acetyl_chloride = Chem.MolFromSmarts("CC(=O)Cl")
        
        has_acetylating_agent = any(
            mol.HasSubstructMatch(acetyl_anhydride) or mol.HasSubstructMatch(acetyl_chloride) 
            for mol in reactants
        )
        
        return has_alcohol_reactant and has_acetylating_agent
    
    def _has_subsequent_chan_lam(self, current_node):
        """Check if Chan-Lam coupling occurs in subsequent reactions"""
        # Look for Chan-Lam coupling patterns in the route tree
        # Chan-Lam typically involves C-N bond formation with Cu catalyst and boronic acid/ester
        
        def search_for_chan_lam(node):
            # Check current node
            if self._is_chan_lam_reaction(node):
                return True
                
            # Recursively search children
            for child in node.get("children", []):
                if search_for_chan_lam(child):
                    return True
            return False
        
        return search_for_chan_lam(current_node)
    
    def _is_chan_lam_reaction(self, node):
        """Identify Chan-Lam N-arylation reaction"""
        metadata = node.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".") if r.strip()]
            product = Chem.MolFromSmiles(rxn_parts[0])
            
            if not product or not reactants:
                return False
            
            # Look for boronic acid/ester pattern (characteristic of Chan-Lam)
            boronic_acid = Chem.MolFromSmarts("[B]([OH])[OH]")
            boronic_ester = Chem.MolFromSmarts("[B]1O[C][C]O1")  # Pinacol boronate
            
            has_boron_reagent = any(
                mol.HasSubstructMatch(boronic_acid) or mol.HasSubstructMatch(boronic_ester)
                for mol in reactants
            )
            
            # Look for amine nucleophile
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactants)
            
            # Look for C-N bond formation (product has N-Ar bond that wasn't in reactants)
            n_aryl_pattern = Chem.MolFromSmarts("N[c]")
            has_n_aryl_product = product.HasSubstructMatch(n_aryl_pattern)
            
            return has_boron_reagent and has_amine and has_n_aryl_product
            
        except Exception:
            return False
