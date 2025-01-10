"""Directly assess a fully generated synthetic route against a given query."""

prefix = """You are an experienced organic chemist tasked with assessing the relevance or similarity of a proposed synthetic route to a given query. You will analyze the reactions carefully, explain the key points of each reaction in relation to the query, and then assess the relevance of the proposed plan for the given query.

First, you will be presented with a query describing a desired synthetic pathway towards a target molecule:

<query>
{query}
</query>

Next, you will be given a sequence of proposed reactions, starting from the target molecule and going backwards through each of the intermediate reactions in a retrosynthetic way.

Each reaction is numbered, and has a depth value indicating its position in the retrosynthetic tree:

<proposed_reactions>
"""

suffix = """</proposed_reactions>
Analyze each reaction in the proposed sequence, starting from the last one (closest to the product) and moving backwards. For each reaction:

1. Identify the key functional groups and structural changes involved.
2. Evaluate how well the reaction aligns with the query's requirements.
3. Discuss any potential issues or improvements.

Write your analysis for each reaction in separate <analysis> tags. Be sure to reference specific aspects of the query when discussing relevance.

After analyzing all reactions, assess the overall relevance of the proposed synthetic route to the query. Consider:

1. How well does the overall sequence align with the query's goals?
2. Are there any major discrepancies or missing steps?

Provide a detailed justification for your assessment, drawing on your analysis of individual reactions and your expertise as an organic chemist.

Finally, assign a relevance score from 0 to 10, where 10 indicates the highest relevance to the query. Present your score in the following format:

<score>[integer from 0 to 10]</score>

Remember, the reactions shown are theoretical and have not been tested in a laboratory. They represent desired transformations but may not necessarily reflect what would actually occur in a flask. Your expertise is crucial in assessing the relevance of these proposed reactions.
"""
