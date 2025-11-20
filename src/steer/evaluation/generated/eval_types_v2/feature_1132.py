"""Generated evaluation code for: Amine to bromide via nitration-reduction sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AmineToBreakSandmeyer(MultiRxnCondBase):
    """
    Checks for amine to bromide conversion via nitration-reduction-Sandmeyer sequence.
    Looks for the presence of NH2 -> NO2 -> NH2 -> Br transformation in aromatic systems.
    """
    
    def __init__(self, config):
        self.require_sequence = config.get("require_sequence", True)
        self.allow_direct = config.get("allow_direct", False)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track functional group transformations
        has_nitration = any(self.detect_nitration(r) for r in reactions)
        has_reduction = any(self.detect_nitro_reduction(r) for r in reactions)
        has_sandmeyer = any(self.detect_sandmeyer_bromination(r) for r in reactions)
        has_direct_amine_br = any(self.detect_direct_amine_bromination(r) for r in reactions)
        
        if self.require_sequence:
            # Must have all three steps of the sequence
            condition = has_nitration and has_reduction and has_sandmeyer
        else:
            # Allow either sequence or direct transformation
            condition = (has_nitration and has_reduction and has_sandmeyer) or \
                       (self.allow_direct and has_direct_amine_br)
        
        return condition, len(reactions)
    
    def detect_nitration(self, rxn):
        """Detect NH2 -> NO2 conversion (nitration)"""
        reactants_smiles, products_smiles = rxn.split(">>")
        
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Look for aromatic amine in reactants
            amine_pattern = Chem.MolFromSmarts("[cH1,c]N([H])[H]")  # Aromatic amine
            has_amine_reactant = any(mol and mol.HasSubstructMatch(amine_pattern) for mol in reactants)
            
            # Look for nitro group in products
            nitro_pattern = Chem.MolFromSmarts("[cH1,c][N+](=O)[O-]")  # Aromatic nitro
            has_nitro_product = any(mol and mol.HasSubstructMatch(nitro_pattern) for mol in products)
            
            return has_amine_reactant and has_nitro_product
            
        except:
            return False
    
    def detect_nitro_reduction(self, rxn):
        """Detect NO2 -> NH2 conversion (reduction)"""
        reactants_smiles, products_smiles = rxn.split(">>")
        
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Look for nitro group in reactants
            nitro_pattern = Chem.MolFromSmarts("[cH1,c][N+](=O)[O-]")
            has_nitro_reactant = any(mol and mol.HasSubstructMatch(nitro_pattern) for mol in reactants)
            
            # Look for aromatic amine in products
            amine_pattern = Chem.MolFromSmarts("[cH1,c]N([H])[H]")
            has_amine_product = any(mol and mol.HasSubstructMatch(amine_pattern) for mol in products)
            
            return has_nitro_reactant and has_amine_product
            
        except:
            return False
    
    def detect_sandmeyer_bromination(self, rxn):
        """Detect NH2 -> Br conversion via Sandmeyer reaction"""
        reactants_smiles, products_smiles = rxn.split(">>")
        
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Look for aromatic amine in reactants
            amine_pattern = Chem.MolFromSmarts("[cH1,c]N([H])[H]")
            has_amine_reactant = any(mol and mol.HasSubstructMatch(amine_pattern) for mol in reactants)
            
            # Look for aromatic bromide in products
            bromide_pattern = Chem.MolFromSmarts("[cH1,c]Br")
            has_bromide_product = any(mol and mol.HasSubstructMatch(bromide_pattern) for mol in products)
            
            # Check for typical Sandmeyer conditions (presence of Cu or CuBr)
            has_copper_reagent = any("Cu" in smi for smi in reactants_smiles.split("."))
            
            return has_amine_reactant and has_bromide_product and has_copper_reagent
            
        except:
            return False
    
    def detect_direct_amine_bromination(self, rxn):
        """Detect direct NH2 -> Br conversion (alternative pathway)"""
        reactants_smiles, products_smiles = rxn.split(">>")
        
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Look for aromatic amine in reactants
            amine_pattern = Chem.MolFromSmarts("[cH1,c]N([H])[H]")
            has_amine_reactant = any(mol and mol.HasSubstructMatch(amine_pattern) for mol in reactants)
            
            # Look for aromatic bromide in products
            bromide_pattern = Chem.MolFromSmarts("[cH1,c]Br")
            has_bromide_product = any(mol and mol.HasSubstructMatch(bromide_pattern) for mol in products)
            
            return has_amine_reactant and has_bromide_product
            
        except:
            return False
