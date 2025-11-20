"""Generated evaluation code for: Sequential deprotection-reprotection of anomeric position"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectionStrategy(MultiRxnCondBase):
    """
    Detects sequential deprotection-reprotection of anomeric position.
    Checks for removal of PMB ether followed by re-alkylation to methyl ether
    in consecutive steps.
    """
    
    def __init__(self, config):
        self.position_smarts = config["position_smarts"]
        self.sequence_type = config["sequence_type"] 
        self.steps_apart = config["steps_apart"]
        self.position_pattern = Chem.MolFromSmarts(self.position_smarts)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find deprotection and reprotection reactions
        deprotection_indices = []
        reprotection_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.is_deprotection(rxn):
                deprotection_indices.append(i)
            elif self.is_reprotection(rxn):
                reprotection_indices.append(i)
        
        # Check for sequential deprotection-reprotection
        condition_met = False
        for deprot_idx in deprotection_indices:
            for reprot_idx in reprotection_indices:
                if abs(reprot_idx - deprot_idx) <= self.steps_apart:
                    if self.same_anomeric_position(reactions[deprot_idx], reactions[reprot_idx]):
                        condition_met = True
                        break
            if condition_met:
                break
        
        return condition_met, len(reactions)
    
    def is_deprotection(self, rxn):
        """Check if reaction removes PMB ether protection"""
        rxn_smiles = rxn.split(">>")
        product = Chem.MolFromSmiles(rxn_smiles[0])
        reactants = [Chem.MolFromSmiles(r) for r in rxn_smiles[1].split(".")]
        
        # Look for PMB group removal (4-methoxybenzyl)
        pmb_pattern = Chem.MolFromSmarts("COc1ccc(C[O])cc1")
        
        # PMB should be in reactants but not in product
        pmb_in_reactants = any(r.HasSubstructMatch(pmb_pattern) for r in reactants if r is not None)
        pmb_in_product = product.HasSubstructMatch(pmb_pattern) if product is not None else False
        
        # Check for anomeric position involvement
        anomeric_in_product = product.HasSubstructMatch(self.position_pattern) if product is not None else False
        
        return pmb_in_reactants and not pmb_in_product and anomeric_in_product
    
    def is_reprotection(self, rxn):
        """Check if reaction adds methyl ether protection"""
        rxn_smiles = rxn.split(">>")
        product = Chem.MolFromSmiles(rxn_smiles[0])
        reactants = [Chem.MolFromSmiles(r) for r in rxn_smiles[1].split(".")]
        
        # Look for methyl ether formation at anomeric position
        methyl_anomeric_pattern = Chem.MolFromSmarts("[CH1]([O][CH3])[O]")
        
        # Methyl ether should be in product but free OH in reactants
        methyl_in_product = product.HasSubstructMatch(methyl_anomeric_pattern) if product is not None else False
        free_oh_in_reactants = any(r.HasSubstructMatch(self.position_pattern) for r in reactants if r is not None)
        
        return methyl_in_product and free_oh_in_reactants
    
    def same_anomeric_position(self, deprotection_rxn, reprotection_rxn):
        """Check if both reactions involve the same anomeric carbon by atom mapping"""
        deprot_smiles = deprotection_rxn.split(">>")
        reprot_smiles = reprotection_rxn.split(">>")
        
        # Get atom map numbers for anomeric carbons
        deprot_product = Chem.MolFromSmiles(deprot_smiles[0])
        reprot_reactant = Chem.MolFromSmiles(reprot_smiles[1].split(".")[0])
        
        if deprot_product is None or reprot_reactant is None:
            return False
        
        # Find anomeric carbon atom map numbers
        deprot_anomeric_maps = self.get_anomeric_atom_maps(deprot_product)
        reprot_anomeric_maps = self.get_anomeric_atom_maps(reprot_reactant)
        
        # Check for overlap in atom map numbers
        return bool(set(deprot_anomeric_maps) & set(reprot_anomeric_maps))
    
    def get_anomeric_atom_maps(self, mol):
        """Get atom map numbers for anomeric carbons in molecule"""
        if mol is None:
            return []
        
        anomeric_maps = []
        matches = mol.GetSubstructMatches(self.position_pattern)
        
        for match in matches:
            anomeric_carbon_idx = match[0]  # First atom in pattern is the anomeric carbon
            anomeric_atom = mol.GetAtomWithIdx(anomeric_carbon_idx)
            if anomeric_atom.GetAtomMapNum() > 0:
                anomeric_maps.append(anomeric_atom.GetAtomMapNum())
        
        return anomeric_maps
