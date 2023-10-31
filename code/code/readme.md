# Target
This repo implements a cross-task molecular GPT model targeting molecular design, reaction prediction and retro-synthesis.

# Motivation
Large language models achieves impressive intelligence in the area of natural language processing. One reason of their success is that NLP tasks share similar forms of data, i.e., sequences. After trained on large data with pre-train self-supervised loss, LLMs are able to learn shared knowledge across tasks. Similarly, molecular tasks can be also formed as sequences with some tricks/designs. 

# Data process
Data from different domains/tasks will be formatted into the same type, as,
<bos> props_embedding , seq !

<eos> -> !
Todo
 - [x] need an eos token
 - [x] how to combine properties and smiles
 - [ ] generate labels