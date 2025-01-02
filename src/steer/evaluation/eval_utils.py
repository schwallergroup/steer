
# For a given synthetic route, define a score. This is query dependent.
# For each query, we define a metric to evaluate the synthetic route.

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from typing import Callable

class DepthCondition:
    """Find out at which depth of the tree a condition is met."""
    def __call__(self, d):
        return self.condition_depth(d['children'][0])+1

    def hit_condition(self, d):
        "Hit condition: define what we are looking for."
        pass

    def condition_depth(self, d, i=0):
        """bfs search for reaction that matches hit condition."""
        if self.hit_condition(d):
            return i
        if 'children' in d:
            for c in d['children']:
                if 'children' in c:
                    a=self.condition_depth(c['children'][0], i+1)
                    if a != -1:
                        return a
        return -2

    def tanimoto(self, smiles1, smiles2):
        # Convert SMILES strings to RDKit molecules
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        # Generate Morgan fingerprints for both molecules
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        # Calculate Tanimoto similarity
        tanimoto_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return tanimoto_similarity
    
    def route_length(self, data):
        """Find the length of the route."""
        # length = [len(d['children']) for d in data]
        # lmscore = [d['lmdata']['routescore'] for d in data]
        pass

    def depth_score(self, data, target_depth: Callable=None):
        """Provide a score based on the depth at which the condition is met."""
        depth = [self(d) for d in data]
        lmscore = [d['lmdata']['routescore'] for d in data]

        # For now let's say difference with target_depth is the score
        score = [target_depth(d) for d in depth]
        return score, lmscore

        
        
