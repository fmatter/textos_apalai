from pyigt.igt import Corpus, IGT
import pandas as pd

corpus = Corpus.from_path("../cldf/metadata_full.json")
# corpus.write_concordance("form", filename="form-concordance.tsv")
# corpus.write_app(dest="concordances")

corpus.check_glosses()