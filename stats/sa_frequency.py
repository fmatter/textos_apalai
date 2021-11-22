import pandas as pd
from pyigt.igt import iter_morphemes, IGT, Corpus
import yaml
from pycldf.terms import Terms
import numpy as np
import scipy
from scipy.stats import chisquare

word_count = 0
morph_count = 0

found_morphemes = {}
test = Corpus.from_path("../cldf/metadata_full.json")
test.check_glosses()
for igt in test:
    for word in igt.glossed_morphemes:
        word_count += 1
        for i, (obj, gloss) in enumerate(word):
            morph_count += 1
            if obj in ["at", "os", "e"] and gloss in ["reflex", "reflx_se", "reflx"]:
                word[i + 1] = list(word[i + 1])
                word[i + 1][0] = obj + "-" + word[i + 1][0]
                word[i + 1][1] = gloss + "-" + word[i + 1][1]
        for obj, gloss in word:
            key = obj.lower() + ":" + gloss.lower()
            if key not in found_morphemes:
                found_morphemes[key] = {"count": 1}
            else:
                found_morphemes[key]["count"] += 1

verbs = yaml.load(open("sa_verbs.json"), Loader=yaml.BaseLoader)
counts = []
for verb in verbs:
    verb_count = 0
    for form in verb["forms"]:
        key = ":".join(form)
        if key in found_morphemes:
            verb_count += found_morphemes[key]["count"]
            found_morphemes[key]["sa"] = True
    counts.append(
        {
            "Form": verb["form"],
            "Meaning": verb["meaning"],
            "Count": verb_count,
        }
    )

df = pd.DataFrame.from_dict(counts)
df["% Sa"] = df["Count"] / sum(df["Count"])
df["% Words"] = df["Count"] / word_count
df.sort_values("Count", ascending=False, inplace=True)
df.to_csv("apalai_sa_verb_stats.csv", index=False)
print(
    df.to_string(
        formatters={
            "% Sa": "{:,.2%}".format,
            "% Words": "{:,.2%}".format,
        }
    )
)

avg = sum(df["Count"]) / len(df)
observed_values = np.array(df["Count"])
expected_values=np.array([avg] * len(df))
a, b = chisquare(observed_values, f_exp=expected_values)
print(a, b)

morphs = []
for morph, data in found_morphemes.items():
    obj, gloss = morph.split(":")
    morphs.append({
        "Form": obj,
        "Gloss": gloss,
        "Count": data["count"],
    })
    if "sa" in data:
        morphs[-1]["Sa_Verb"] = data["sa"]
w_df = pd.DataFrame.from_dict(morphs)
w_df.sort_values("Count", ascending=False, inplace=True)
w_df.fillna(False, inplace=True)
w_df.to_csv("all_morphemes.csv", index=False)