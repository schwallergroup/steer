"""Generated evaluation code for: N-Boc protection of amide nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NBocAmideProtection(BaseScoring):
    """
    Detects N-Boc protection of amide nitrogen in synthesis routes.
    
    This class identifies reactions where a Boc protecting group is added to 
    an amide nitrogen, which is typically chemically challenging and 
    strategically questionable due to the reduced nucleophilicity of amide nitrogens.
    """
    
    def __init__(self, config: Dict):
        self.present = config.get("present", True)
        # SMARTS pattern for N-Boc protected amide
        # [N;$(N-C(=O)-*)] matches amide nitrogen
        # C(=O)OC(C)(C)C matches Boc group
        self.nboc_amide_pattern = "[N;$(N-C(=O)-*)]C(=O)OC(C)(C)C"
        
    def route_scoring(self, x) -> float:
        """Convert depth to score. Earlier protection is worse (higher penalty)."""
        if x < 0:  # Condition not met
            return 0 if self.present else 10
        else:  # Condition met at depth x
            if self.present:
                # Penalize early N-Boc protection more heavily
                return 10 * (1 - x)  # Earlier = higher penalty
            else:
                return 10 * x  # Penalize if found when not expected
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves N-Boc protection of an amide nitrogen."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactants_mol = Chem.MolFromSmiles(reactants_smiles)
            products_mol = Chem.MolFromSmiles(products_smiles)
            
            if not reactants_mol or not products_mol:
                return False
            
            # Check if product has N-Boc amide that reactant doesn't have
            reactant_has_nboc = reactants_mol.HasSubstructMatch(
                Chem.MolFromSmarts(self.nboc_amide_pattern)
            )
            product_has_nboc = products_mol.HasSubstructMatch(
                Chem.MolFromSmarts(self.nboc_amide_pattern)
            )
            
            # Also check for free amide nitrogen in reactants
            free_amide_pattern = "[N;$(N-C(=O)-*);!$(NC(=O)OC(C)(C)C)]"
            reactant_has_free_amide = reactants_mol.HasSubstructMatch(
                Chem.MolFromSmarts(free_amide_pattern)
            )
            
            # N-Boc protection occurs when:
            # 1. Reactant has free amide nitrogen
            # 2. Product has N-Boc amide
            # 3. Reactant doesn't already have N-Boc amide
            return (reactant_has_free_amide and 
                   product_has_nboc and 
                   not reactant_has_nboc)
                   
        except Exception:
            return False
