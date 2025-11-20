"""Generated evaluation code for: Convergent synthesis via separate hydrazine and dicarbonyl fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentHydrazineDicarbonyl(BaseScoring):
    """
    Evaluates convergent synthesis strategy involving separate preparation of 
    hydrazine and beta-dicarbonyl fragments followed by heterocyclic cyclization.
    
    Detects reactions where:
    1. An aryl hydrazine fragment (containing N-NH2 or N-NHR)
    2. A beta-dicarbonyl fragment (containing C(=O)-C-C(=O) motif)
    3. Are coupled in a cyclization reaction to form heterocycles
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Earlier convergent coupling is better (closer to 0)
            if self.condition_type == "bool":
                return 1  # Found the coupling
            else:
                return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """Check if this reaction represents convergent hydrazine-dicarbonyl coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        # Parse reactants
        reactant_parts = reactant_smiles.split(".")
        if len(reactant_parts) < 2:
            return False  # Need at least 2 fragments for convergent synthesis
            
        reactants = []
        for r_smiles in reactant_parts:
            mol = Chem.MolFromSmiles(r_smiles)
            if mol:
                reactants.append(mol)
        
        if len(reactants) < 2:
            return False
            
        # Check for product containing heterocycle
        product = Chem.MolFromSmiles(product_smiles)
        if not product or not self._has_heterocycle(product):
            return False
            
        # Check reactants for hydrazine and dicarbonyl fragments
        has_hydrazine = False
        has_dicarbonyl = False
        
        for reactant in reactants:
            if self._has_aryl_hydrazine(reactant):
                has_hydrazine = True
            elif self._has_beta_dicarbonyl(reactant):
                has_dicarbonyl = True
                
        return has_hydrazine and has_dicarbonyl
    
    def _has_aryl_hydrazine(self, mol):
        """Check if molecule contains aryl hydrazine motif"""
        # Aryl hydrazine patterns: Ar-NH-NH2, Ar-NH-NHR
        patterns = [
            "[cH0,cH1:1]-[NH1:2]-[NH2:3]",  # Ar-NH-NH2
            "[cH0,cH1:1]-[NH1:2]-[NH1:3]-[C,c:4]",  # Ar-NH-NHR
            "[cH0,cH1:1]-[N:2](-[NH2:3])-[C,c:4]"  # Ar-N(NH2)-R
        ]
        
        for pattern in patterns:
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and mol.HasSubstructMatch(patt_mol):
                return True
        return False
    
    def _has_beta_dicarbonyl(self, mol):
        """Check if molecule contains beta-dicarbonyl motif"""
        # Beta-dicarbonyl patterns: R-CO-CH2-CO-R, R-CO-CHR-CO-R
        patterns = [
            "[C,c:1]-[C:2](=[O:3])-[CH2:4]-[C:5](=[O:6])-[C,c:7]",  # R-CO-CH2-CO-R
            "[C,c:1]-[C:2](=[O:3])-[CH1:4](-[C,c:8])-[C:5](=[O:6])-[C,c:7]",  # R-CO-CHR-CO-R
            "[C,c:1]-[C:2](=[O:3])-[CH1:4](-[C:8])-[C:5](=[O:6])-[C,c:7]"  # Including aliphatic substituents
        ]
        
        for pattern in patterns:
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and mol.HasSubstructMatch(patt_mol):
                return True
        return False
    
    def _has_heterocycle(self, mol):
        """Check if molecule contains nitrogen heterocycle (common product of this coupling)"""
        # Common heterocycles formed: pyrazoles, pyridazines, indazoles
        patterns = [
            "[nH0,nH1:1]1-[c,n:2]-[c,n:3]-[c,n:4]-[c,n:5]-1",  # 6-membered N-heterocycle
            "[nH0,nH1:1]1-[c,n:2]-[c,n:3]-[c,n:4]-1",  # 5-membered N-heterocycle
            "[nH0,nH1:1]1-[nH0,nH1:2]-[c:3]-[c:4]-[c:5]-1"  # N-N containing heterocycle
        ]
        
        for pattern in patterns:
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and mol.HasSubstructMatch(patt_mol):
                return True
        return False
