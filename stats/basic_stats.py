from pyigt import Corpus
import pandas as pd
word_count = 0
morph_count = 0

corpus = Corpus.from_path("../cldf/metadata_full.json")
# corpus.write_concordance("form", filename="form-concordance.tsv")
# corpus.write_app(dest="concordances")

f = open("stats.md", "w")
for name, value in zip(["Examples", "Words", "Morphemes"], corpus.get_stats()):
    f.write(f"* {name}: {value}\n")
f.close()

corpus.check_glosses()