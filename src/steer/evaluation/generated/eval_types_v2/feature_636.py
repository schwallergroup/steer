"""Generated evaluation code for: Evans auxiliary stereoselective alkylation strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EvansAuxiliaryAlkylation(BaseScoring):
    """
    Evaluates synthesis routes for the use of Evans chiral auxiliary in stereoselective 
    alkylation reactions. Detects the presence of oxazolidinone auxiliary and checks
    if it's used in mid-stage C-C bond forming reactions.
    """
    
    def __init__(self, config: Dict):
        self.auxiliary_type = config.get("auxiliary_type", "evans_oxazolidinone")
        self.purpose = config.get("purpose", "stereocontrol")
        self.timing = config.get("timing", "mid_stage")
        
        # Evans auxiliary oxazolidinone pattern
        self.oxazolidinone_pattern = Chem.MolFromSmarts("[#6]1[#6][#7][#6](=[#8])[#8]1")
        # Chiral benzyl oxazolidinone (common Evans auxiliary)
        self.evans_auxiliary_pattern = Chem.MolFromSmarts("c1ccc(cc1)[CH]2[CH2][NH][C](=O)[O]2")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        
        # For mid-stage timing, prefer reactions around 30-70% through synthesis
        if self.timing == "mid_stage":
            if 0.2 <= x <= 0.8:
                return 8 + 2 * (1 - abs(x - 0.5) * 2)  # Peak score at x=0.5
            elif x < 0.2:
                return 3 + 5 * (x / 0.2)  # Penalty for too early
            else:
                return 3 + 5 * ((1 - x) / 0.2)  # Penalty for too late
        elif self.timing == "early_stage":
            return 8 + 2 * (1 - x)  # Earlier is better
        else:  # late_stage
            return 8 + 2 * x  # Later is better
    
    def hit_condition(self, d):
        """
        Check if this reaction involves Evans auxiliary alkylation
        """
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
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            if not prod_mol:
                return False
                
            reactant_mols = []
            for r_smi in reactants.split("."):
                r_mol = Chem.MolFromSmiles(r_smi)
                if r_mol:
                    reactant_mols.append(r_mol)
            
            if not reactant_mols:
                return False
            
            # Check for Evans auxiliary presence in reactants
            has_evans_auxiliary = any(
                self._has_evans_auxiliary(mol) for mol in reactant_mols
            )
            
            if not has_evans_auxiliary:
                return False
            
            # Check if this is an alkylation reaction (C-C bond formation)
            if not self._is_alkylation_reaction(reactant_mols, prod_mol):
                return False
                
            # Check if auxiliary is used for stereocontrol
            if self.purpose == "stereocontrol":
                return self._involves_stereocontrol(reactant_mols, prod_mol)
            
            return True
            
        except Exception:
            return False
    
    def _has_evans_auxiliary(self, mol):
        """Check if molecule contains Evans auxiliary motif"""
        if not mol:
            return False
            
        # Check for general oxazolidinone pattern first
        if not mol.HasSubstructMatch(self.oxazolidinone_pattern):
            return False
            
        # Check for specific Evans auxiliary pattern (benzyl oxazolidinone)
        if mol.HasSubstructMatch(self.evans_auxiliary_pattern):
            return True
            
        # Also check for other common Evans auxiliaries (isopropyl, etc.)
        # Oxazolidinone with branched alkyl substituent
        branched_evans = Chem.MolFromSmarts("[CH]([CH3])[CH3][CH]1[CH2][NH][C](=O)[O]1")
        if mol.HasSubstructMatch(branched_evans):
            return True
            
        return False
    
    def _is_alkylation_reaction(self, reactants, product):
        """Check if this is a C-C bond forming alkylation reaction"""
        if not reactants or not product:
            return False
            
        # Count carbon atoms in reactants vs product
        reactant_carbons = sum(
            sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            for mol in reactants
        )
        product_carbons = sum(1 for atom in product.GetAtoms() if atom.GetAtomicNum() == 6)
        
        # Should have same number of carbons (intramolecular alkylation)
        # or slight increase (if small alkylating agent)
        carbon_diff = product_carbons - reactant_carbons
        
        # Look for patterns suggesting alkylation
        # Alkyl halide pattern
        alkyl_halide = Chem.MolFromSmarts("[CH3,CH2,CH][Cl,Br,I]")
        # Alkyl tosylate/mesylate pattern  
        alkyl_sulfonate = Chem.MolFromSmarts("[CH3,CH2,CH]OS(=O)(=O)")
        
        has_alkylating_agent = any(
            mol.HasSubstructMatch(alkyl_halide) or mol.HasSubstructMatch(alkyl_sulfonate)
            for mol in reactants
        )
        
        return has_alkylating_agent or (-2 <= carbon_diff <= 4)
    
    def _involves_stereocontrol(self, reactants, product):
        """Check if the reaction likely involves stereocontrol"""
        # Look for chiral centers being formed
        # This is a simplified check - in practice would need more sophisticated analysis
        
        # Check for increase in chiral centers
        reactant_chiral_centers = sum(
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            for mol in reactants if mol
        )
        
        product_chiral_centers = len(Chem.FindMolChiralCenters(product, includeUnassigned=True))
        
        # If new chiral centers are formed, likely stereocontrol
        if product_chiral_centers > reactant_chiral_centers:
            return True
            
        # Also check for alpha-alkylation pattern (common with Evans auxiliary)
        alpha_carbon_pattern = Chem.MolFromSmarts("[CH,CH2][C](=O)[NH][C](=O)[O]")
        
        return any(mol.HasSubstructMatch(alpha_carbon_pattern) for mol in reactants if mol)
