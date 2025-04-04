"""Directly assess a partially generated synthetic route against a given query."""

# prefix = """You are an experienced organic chemist tasked with assessing the relevance or similarity of a proposed synthetic route to a given query. You will analyze the reactions carefully, explain the key points of each reaction in relation to the query, and then assess the relevance of the proposed plan for the given query.

# First, the following query describes a desired synthetic pathway towards a target molecule:

# <query>
# {query}
# </query>

# Next, you will be given a sequence of proposed reactions, starting from the target molecule and going backwards through each of the intermediate reactions in a retrosynthetic way.
# Note that:

# 1. "Early" in the synthesis means further from the target molecule, as the reactions further back in the sequence are closer to the starting materials.
# 2. "Late" and "late-stage" means closer to the target molecule.
# 3. "Break" indicates a retrosynthetic step, where a molecule is broken down into simpler components. In the forward direction, this would mean "Form". For example, "Break C-C bond" would be equivalent to "Form C-C bond" in the forward direction.

# Each reaction is numbered, and has a depth value indicating its position in the retrosynthetic tree.
# Furthermore, the synthetic routes might not be fully solved, which means that some steps can still be completed. Your task is to assess the potential of the current route to fulfill the query's requirements. However, the reactions shown so far will not be changed, only further reactions might be added, so consider this.

# <proposed_reactions>
# """

# suffix = """</proposed_reactions>
# Analyze each reaction in the proposed sequence, starting from the last one (closest to the product) and moving backwards. For each reaction:

# 1. Identify the key functional groups and structural changes involved.
# 2. Evaluate how well the reaction aligns with the query's requirements.

# Write your analysis for each reaction in separate <analysis> tags. Be sure to reference specific aspects of the query when discussing relevance.

# After analyzing all reactions, assess the overall relevance of the proposed synthetic route to the query. Consider:

# 1. How well does the overall sequence align with the query's goals?
# 2. Are there any major discrepancies or missing steps?

# Provide a detailed justification for your assessment, drawing on your analysis of individual reactions and your expertise as an organic chemist.

# Finally, assign a relevance score from 0 to 10, where 10 indicates the highest relevance to the query. Present your score in the following format:

# <score>[integer from 0 to 10]</score>

# Remember, the reactions shown are theoretical and have not been tested in a laboratory. They represent desired transformations but may not necessarily reflect what would actually occur in a flask. Your expertise is crucial in assessing the relevance of these proposed reactions.
# """


prefix = """You are an expert chemist assisting with a retrosynthesis planning program. Your task is to evaluate a partially constructed synthetic route and assess its potential alignment with a user's specific request.

The user has provided the following query describing their desired synthetic route:
<user_query>
{query}
</user_query>

Next, you will be given a sequence of proposed reactions, starting from the target molecule and going backwards through each of the intermediate reactions in a retrosynthetic way.
Note that:

1. "Early" in the synthesis means further from the target molecule, as the reactions further back in the sequence are closer to the starting materials.
2. "Late" and "late-stage" means closer to the target molecule.
3. "Break" indicates a retrosynthetic step, where a molecule is broken down into simpler components. In the forward direction, this would mean "Form". For example, "Break C-C bond" would be equivalent to "Form C-C bond" in the forward direction.

Here is the partially constructed route:
<proposed_reactions>"""

suffix = """</proposed_reactions>

Please follow these steps:

1. Carefully analyze the proposed reactions. Consider the types of reactions, reagents used, and the overall strategy of the partial route.

2. Examine the user's query in detail. Identify key requirements, preferences, or constraints mentioned by the user regarding the desired synthetic route.

3. Assess the potential for the current partially constructed route to align with the user's request. Consider how well the proposed reactions match the user's described strategy, preferred reaction types, or other specified criteria.

4. Provide a detailed justification for your assessment. Explain which aspects of the partial route align well with the user's request and which aspects may not meet their criteria. Be specific in your reasoning.

5. Based on your analysis, assign a score from 0 to 10, where 10 indicates perfect alignment with the user's request and 0 indicates no alignment at all.

Present your evaluation in the following format:
<evaluation>
<justification>
[Your detailed justification here]
</justification>
<score>[Your numerical score from 0 to 10]</score>
</evaluation>"""
