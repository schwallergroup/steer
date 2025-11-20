"""Generated evaluation code for: Late stage reductive amination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageReductiveAmination(BaseScoring):
    """
    Evaluates whether a reductive amination reaction occurs late in the synthesis route.
    Returns higher scores when reductive amination happens after the specified stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"]["stage_threshold"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reductive amination doesn't occur
        elif x >= self.stage_threshold:
            return 10  # Perfect score for very late stage
        else:
            # Linear scaling: later is better, but penalize if before threshold
            return max(0, (x - 0.5) * 20)  # Scale to 0-10 range
    
    def hit_condition(self, d) -> bool:
        """
        Detect reductive amination by checking for:
        1. Formation of C-N bond where nitrogen was previously part of imine/enamine
        2. Presence of reducing agents (NaBH4, NaBH3CN, etc.)
        3. Conversion of C=N to C-N pattern
        """
        metadata = d.get("metadata", {})
        
        # Check for reductive amination reagents
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        # Common reductive amination reagents patterns
        reducing_agents = [
            "[BH4-]",  # Borohydride
            "[BH3-]",  # Cyanoborohydride  
            "B(C)H",   # Other borane reagents
        ]
        
        reagent_present = any(agent in rxn_smiles for agent in reducing_agents)
        
        # Check for imine/enamine reduction pattern
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        reactants = parts[0]
        products = parts[1]
        
        try:
            # Look for C=N pattern in reactants and C-N in products
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".") if Chem.MolFromSmiles(r)]
            product_mols = [Chem.MolFromSmiles(p) for p in products.split(".") if Chem.MolFromSmiles(p)]
            
            # Imine/enamine patterns
            imine_pattern = Chem.MolFromSmarts("[C]=[N]")
            enamine_pattern = Chem.MolFromSmarts("[C]=[C]-[N]")
            
            # Secondary/tertiary amine pattern  
            amine_pattern = Chem.MolFromSmarts("[C]-[N]([C,H])[C,H]")
            
            has_imine_reactant = any(mol.HasSubstructMatch(imine_pattern) or 
                                   mol.HasSubstructMatch(enamine_pattern) 
                                   for mol in reactant_mols if mol)
            
            has_amine_product = any(mol.HasSubstructMatch(amine_pattern) 
                                  for mol in product_mols if mol)
            
            return (reagent_present or has_imine_reactant) and has_amine_product
            
        except:
            return False
