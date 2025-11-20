"""Generated evaluation code for: Late stage protecting group installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageProtectingGroupInstallation(BaseScoring):
    """
    Evaluates whether a specific protecting group is installed late in the synthesis.
    Penalizes routes where protecting groups are added too early when they could
    be added closer to the end of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.timing = config["parameters"]["timing"]
        self.functional_group = config["parameters"]["functional_group"]
        self.step_position = config["parameters"]["step_position"]
        
        # Define SMARTS patterns for common protecting groups
        self.protecting_group_patterns = {
            "SEM": "[CH2][Si]([CH3])([CH3])[CH2][O][CH3]",  # SEM protecting group
            "Boc": "[N][C](=O)[O][C]([CH3])([CH3])[CH3]",   # Boc protecting group
            "Cbz": "[N][C](=O)[O][CH2]c1ccccc1",            # Cbz protecting group
            "TBS": "[O][Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]",  # TBS protecting group
            "Bn": "[N][CH2]c1ccccc1",                       # Benzyl protecting group
        }
        
        # Define functional group patterns
        self.functional_group_patterns = {
            "NH": "[NH]",      # Primary amine
            "NH2": "[NH2]",    # Primary amine (explicit)
            "OH": "[OH]",      # Hydroxyl group
            "SH": "[SH]",      # Thiol group
        }

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protecting group installation doesn't happen
        
        if self.timing == "late":
            # For late-stage installation, lower depth fraction is better
            # x is depth fraction (0 = at target, 1 = at root)
            if x <= (self.step_position / 10.0):  # Within acceptable late-stage window
                return 10 * (1 - x)  # Reward later installation
            else:
                return 5 * (1 - x)   # Penalize early installation
        else:
            # For early-stage installation, higher depth fraction is better
            return 10 * x

    def hit_condition(self, d) -> bool:
        """
        Detect if this reaction involves installation of the target protecting group
        on the specified functional group.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            products = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not products or not all(reactants):
                return False
            
            # Check if protecting group appears in products but not in main reactant
            pg_pattern = self.protecting_group_patterns.get(self.protecting_group)
            fg_pattern = self.functional_group_patterns.get(self.functional_group)
            
            if not pg_pattern or not fg_pattern:
                return False
                
            pg_smarts = Chem.MolFromSmarts(pg_pattern)
            fg_smarts = Chem.MolFromSmarts(fg_pattern)
            
            if not pg_smarts or not fg_smarts:
                return False
            
            # Check if product has the protecting group
            has_pg_in_product = products.HasSubstructMatch(pg_smarts)
            
            # Check if the main reactant (first/largest) lacks the protecting group
            # but has the target functional group
            main_reactant = max(reactants, key=lambda x: x.GetNumAtoms() if x else 0)
            
            has_fg_in_reactant = main_reactant.HasSubstructMatch(fg_smarts)
            has_pg_in_reactant = main_reactant.HasSubstructMatch(pg_smarts)
            
            # This is a protecting group installation if:
            # 1. Product has the protecting group
            # 2. Main reactant has the functional group but not the protecting group
            # 3. At least one reactant contains the protecting group precursor
            is_pg_installation = (has_pg_in_product and 
                                has_fg_in_reactant and 
                                not has_pg_in_reactant and
                                any(r.HasSubstructMatch(pg_smarts) or 
                                    self._is_pg_precursor(r) for r in reactants))
            
            return is_pg_installation
            
        except Exception:
            return False
    
    def _is_pg_precursor(self, mol) -> bool:
        """Check if molecule is likely a protecting group reagent/precursor."""
        if not mol:
            return False
            
        # Common protecting group reagent patterns
        precursor_patterns = {
            "SEM": ["[Cl][CH2][Si]", "[Br][CH2][Si]"],  # SEM-Cl, SEM-Br
            "Boc": ["[C](=O)[O][C]([CH3])([CH3])[CH3]"], # Boc2O, Boc-Cl
            "Cbz": ["[Cl][C](=O)[O][CH2]c1ccccc1"],      # Cbz-Cl
            "TBS": ["[Cl][Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]"], # TBS-Cl
            "Bn": ["[Br][CH2]c1ccccc1", "[Cl][CH2]c1ccccc1"], # Benzyl halides
        }
        
        patterns = precursor_patterns.get(self.protecting_group, [])
        for pattern in patterns:
            try:
                smarts = Chem.MolFromSmarts(pattern)
                if smarts and mol.HasSubstructMatch(smarts):
                    return True
            except:
                continue
                
        return False
