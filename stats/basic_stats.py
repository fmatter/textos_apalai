from pyigt import Corpus
import pandas as pd
word_count = 0
morph_count = 0

corpus = Corpus.from_path("../cldf/cldf-metadata.json")
# corpus.write_concordance("form", filename="form-concordance.tsv")
# corpus.write_app(dest="concordances")

f = open("stats.md", "w")
for name, value in zip(["Examples", "Words", "Morphemes"], corpus.get_stats()):
    f.write(f"* {name}: {value}\n")
f.close()

corpus.check_glosses()

df = pd.read_csv("../cldf/examples.csv")

def print_example(id):
    res = df[df["Example_ID"] == id].iloc[0]
    print(res["Primary_Text"])
    for a, b in zip(res["Analyzed_Word"].split("\t"), res["Gloss"].split("\t")):
        print(a, b)
    print(res["Translated_Text"])
