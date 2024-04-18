import spacy

# load the en_core_web_sm pipeline
nlp = spacy.load("en_core_web_sm")

# print the names of the pipeline components
print(nlp.pipe_names, "\n")

# print the full pipeline of (name, component tuples)
print(nlp.pipeline, "\n")








