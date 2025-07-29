### ai-hallucination-detection
 AI Hallucination Detection
## Overview:
This project explores how often and why AI language models hallucinate, meaning they generate incorrect, misleading, or fabricated information especially in response to complex or rare factual prompts. Understanding and detecting hallucinations is critical to ensure AI systems remain trustworthy in high-risk fields where there is little room for error.

## Objectives:
Measure hallucination rates across various different pre-made as well as custom made questions.

Analyze patterns in hallucination behavior.

Explore preliminary methods to predict hallucinations based on AI output characteristics.

## Methods:
Prompt a language model with a set of easy and hard factual questions.

Collect and fact-check the AI responses.

Analyze hallucination frequency, severity, and topic sensitivity.

Identify keywords that the AI tends to mess up on and use a Pytorch model to extract important features, then graph results. 

## Future Work:
Automating hallucination detection using uncertainty estimation or output features.

Testing additional AI models and comparing results.

Applying techniques in high-risk fields (e.g., medical or legal AI applications).
