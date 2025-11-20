"""Generated evaluation code for: Benzyl protecting group for phenol strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylPhenolProtection(BaseScoring):
    """
    Evaluates benzyl protecting group strategy for phenol groups.
    Checks for installation of benzyl ether protection and its removal by hydrogenolysis.
    """
    
    def __init__(self, config: Dict):
        self.installation_step = config["parameters"]["installation_step"]
        self.removal_step = config["parameters"]["removal_step"]
        self.current_search = "installation"  # Track which step we're looking for
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        
        # Convert depth to step number and evaluate timing
        if self.current_search == "installation":
            # Early installation is preferred
            step_penalty = abs(x * 20 - self.installation_step) / 20
            return max(0, 1 - step_penalty)
        else:  # removal
            # Late removal is preferred  
            step_penalty = abs(x * 20 - self.removal_step) / 20
            return max(0, 1 - step_penalty)
    
    def hit_condition(self, d):
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        reactants = parts[0]
        products = parts[1]
        
        if self.current_search == "installation":
            return self._detect_benzyl_installation(reactants, products)
        else:
            return self._detect_benzyl_removal(reactants, products)
    
    def _detect_benzyl_installation(self, reactants, products):
        """Detect benzyl ether formation from phenol + benzyl halide/alcohol"""
        # Phenol pattern
        phenol_pattern = Chem.MolFromSmarts("[OH1][c]")
        # Benzyl ether pattern  
        benzyl_ether_pattern = Chem.MolFromSmarts("[CH2]([c])[O][c]")
        # Benzyl halide/alcohol patterns
        benzyl_halide_pattern = Chem.MolFromSmarts("[CH2]([c])[Cl,Br,I]")
        benzyl_alcohol_pattern = Chem.MolFromSmarts("[CH2]([c])[OH]")
        
        try:
            # Check reactants for phenol and benzyl compound
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            has_phenol = any(mol and mol.HasSubstructMatch(phenol_pattern) for mol in reactant_mols)
            has_benzyl_reagent = any(mol and (mol.HasSubstructMatch(benzyl_halide_pattern) or 
                                            mol.HasSubstructMatch(benzyl_alcohol_pattern)) 
                                   for mol in reactant_mols)
            has_benzyl_ether = any(mol and mol.HasSubstructMatch(benzyl_ether_pattern) for mol in product_mols)
            
            return has_phenol and has_benzyl_reagent and has_benzyl_ether
            
        except:
            return False
    
    def _detect_benzyl_removal(self, reactants, products):
        """Detect benzyl ether cleavage by hydrogenolysis"""
        # Benzyl ether pattern
        benzyl_ether_pattern = Chem.MolFromSmarts("[CH2]([c])[O][c]")
        # Phenol pattern
        phenol_pattern = Chem.MolFromSmarts("[OH1][c]")
        # Toluene pattern (hydrogenolysis byproduct)
        toluene_pattern = Chem.MolFromSmarts("[CH3][c]1[cH][cH][cH][cH][cH]1")
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            has_benzyl_ether_reactant = any(mol and mol.HasSubstructMatch(benzyl_ether_pattern) 
                                          for mol in reactant_mols)
            has_phenol_product = any(mol and mol.HasSubstructMatch(phenol_pattern) 
                                   for mol in product_mols)
            has_toluene_product = any(mol and mol.HasSubstructMatch(toluene_pattern) 
                                    for mol in product_mols)
            
            return has_benzyl_ether_reactant and has_phenol_product and has_toluene_product
            
        except:
            return False
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Override to check both installation and removal steps"""
        # First check for installation
        self.current_search = "installation"
        installation_found, installation_depth = super().condition_depth(d)
        
        # Then check for removal
        self.current_search = "removal"  
        removal_found, removal_depth = super().condition_depth(d)
        
        # Strategy is complete if both steps are found
        strategy_complete = installation_found and removal_found
        
        # Return combined score based on both steps
        if strategy_complete:
            combined_depth = (installation_depth + removal_depth) / 2
            return True, combined_depth
        else:
            return False, -1
