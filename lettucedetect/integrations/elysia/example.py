"""Example of using LettuceDetect with Elysia for automatic hallucination detection."""

from elysia import Tree

from lettucedetect.integrations.elysia import detect_hallucinations

# Create an Elysia tree with hallucination detection capabilities
tree = Tree()

# Add LettuceDetect tools to the tree
tree.add_tool(detect_hallucinations)

tree(
    "How many data they generated in Kovacs et al. 2025? Please answer and verify your response for credibility."
)
