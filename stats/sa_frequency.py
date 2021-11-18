import pandas as pd
from pyigt.igt import iter_morphemes, IGT, Corpus
import yaml
from pycldf.terms import Terms


def check_gloss(obj, gloss):
    igt = IGT("nil", phrase = obj.split(" "), gloss=gloss.split(" "), properties={})
    if not igt.is_valid():
        print(igt)
    return igt.glossed_morphemes

word_count = 0
morph_count = 0

found_morphemes = {}
test = Corpus.from_path("koehn_a_koehn_apalai/cldf/cldf-metadata.json")
for igt in test:
    for word in igt.glossed_morphemes:
        word_count += 1
        for i, (obj, gloss) in enumerate(word):
            morph_count += 1
            if obj in ["at"] and gloss in ["reflex"]:
                word[i+1] = list(word[i+1])
                word[i+1][0] = obj + "-" + word[i+1][0]
                word[i+1][1] = gloss + "-" + word[i+1][1]
        for obj, gloss in word:
            key = obj.lower()+":"+gloss.lower()
            if key not in found_morphemes:
                found_morphemes[key] = 1
            else:
                found_morphemes[key] += 1

verbs = yaml.load(open("sa_verbs.json"))
counts = []
for verb in verbs:
    verb_count = 0
    for coin in verb["coins"]:
        key = ":".join(coin)
        if key in found_morphemes:
            verb_count += found_morphemes[key]
    counts.append({
        "count": verb_count,
        "verb": verb["meaning"],
        "forms": ", ".join(set(x[0] for x in verb["coins"])),
    })
            
df = pd.DataFrame.from_dict(counts)

df["%\m"] = df["count"] / morph_count
df["%\w"] = df["count"] / word_count
print(df.sort_values("count", ascending=False).to_string(formatters={
    "%\w": '{:,.2%}'.format,
    "%\m": '{:,.2%}'.format,
}))
