"""Generated evaluation code for: Nitro group dual functionality strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitroDualFunctionality(MultiRxnCondBase):
    """
    Evaluates synthesis routes for nitro group dual functionality strategy.
    
    Checks if nitro groups are used both as:
    1. Activating groups (e.g., in SNAr reactions)
    2. Amine precursors (through reduction to form cyclization substrates)
    
    Rewards routes that demonstrate sequential use of nitro group functionality.
    """
    
    def __init__(self, config):
        self.require_sequential = config.get("sequential_use", True)
        self.min_uses = config.get("min_dual_uses", 1)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        nitro_activating_reactions = []
        nitro_reduction_reactions = []
        
        # Identify reactions where nitro acts as activating group
        for i, rxn in enumerate(reactions):
            if self.detect_nitro_activation(rxn):
                nitro_activating_reactions.append(i)
                
        # Identify reactions where nitro is reduced to amine
        for i, rxn in enumerate(reactions):
            if self.detect_nitro_reduction(rxn):
                nitro_reduction_reactions.append(i)
        
        # Check if both functionalities are present
        has_dual_functionality = len(nitro_activating_reactions) > 0 and len(nitro_reduction_reactions) > 0
        
        # If sequential use is required, check order
        if self.require_sequential and has_dual_functionality:
            # Activation should generally occur before reduction
            earliest_activation = min(nitro_activating_reactions) if nitro_activating_reactions else float('inf')
            latest_reduction = max(nitro_reduction_reactions) if nitro_reduction_reactions else -1
            sequential_condition = earliest_activation < latest_reduction
        else:
            sequential_condition = True
            
        condition_met = has_dual_functionality and sequential_condition
        
        return condition_met, len(reactions)
    
    def detect_nitro_activation(self, rxn):
        """
        Detects if nitro group acts as activating group in SNAr or similar reactions.
        Looks for nitro-substituted aromatics as reactants with nucleophilic substitution patterns.
        """
        prod_mol, react_mols = self.parse_reaction_smiles(rxn)
        
        # Nitro-substituted aromatic pattern
        nitro_aromatic_pattern = Chem.MolFromSmarts("[cH0,c:1][N+](=O)[O-]")
        
        # Check if reactants contain nitro-substituted aromatics
        for react_mol in react_mols:
            if react_mol and react_mol.HasSubstructMatch(nitro_aromatic_pattern):
                # Check if this appears to be a substitution reaction
                # by comparing atom counts between reactants and products
                if self.is_substitution_pattern(react_mol, prod_mol):
                    return True
                    
        return False
    
    def detect_nitro_reduction(self, rxn):
        """
        Detects reduction of nitro group to amine.
        Looks for nitro in reactants and corresponding amine in products.
        """
        prod_mol, react_mols = react_mols_list = self.parse_reaction_smiles(rxn)
        
        # Nitro group pattern
        nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
        # Amine patterns (primary, secondary)
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        
        # Check for nitro in reactants
        has_nitro_reactant = any(
            mol and mol.HasSubstructMatch(nitro_pattern) 
            for mol in react_mols if mol
        )
        
        # Check for amine in products
        has_amine_product = prod_mol and prod_mol.HasSubstructMatch(amine_pattern)
        
        # Additional check: ensure the carbon framework is preserved
        if has_nitro_reactant and has_amine_product:
            return self.verify_nitro_to_amine_transformation(react_mols, prod_mol)
            
        return False
    
    def is_substitution_pattern(self, reactant, product):
        """
        Checks if the reaction pattern suggests nucleophilic aromatic substitution.
        """
        if not (reactant and product):
            return False
            
        # Simple heuristic: check if aromatic carbon count is preserved
        # but heteroatom composition changes (suggesting substitution)
        react_aromatic_c = len([a for a in reactant.GetAtoms() 
                               if a.GetIsAromatic() and a.GetSymbol() == 'C'])
        prod_aromatic_c = len([a for a in product.GetAtoms() 
                              if a.GetIsAromatic() and a.GetSymbol() == 'C'])
        
        return react_aromatic_c == prod_aromatic_c
    
    def verify_nitro_to_amine_transformation(self, reactants, product):
        """
        Verifies that nitro reduction to amine occurred by checking atom mapping
        or structural similarity.
        """
        if not product:
            return False
            
        for reactant in reactants:
            if not reactant:
                continue
                
            # Count heavy atoms to ensure similar molecular framework
            react_heavy = reactant.GetNumHeavyAtoms()
            prod_heavy = product.GetNumHeavyAtoms()
            
            # Nitro to amine should reduce heavy atom count by 2 (loss of 2 O atoms)
            if react_heavy - prod_heavy == 2:
                return True
                
        return False
    
    def parse_reaction_smiles(self, rxn_smiles):
        """
        Parse reaction SMILES to get product and reactant molecules.
        """
        try:
            if ">>" in rxn_smiles:
                reactant_smiles, product_smiles = rxn_smiles.split(">>")
            else:
                # Assume it's stored in metadata
                return None, []
                
            product = Chem.MolFromSmiles(product_smiles.strip())
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            return product, reactants
        except:
            return None, []
