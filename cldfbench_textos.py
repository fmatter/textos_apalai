import pathlib
from tika import parser
import os
import subprocess
import json
import pandas as pd
from segments import Tokenizer, Profile
import unicodedata
from cldfbench import CLDFSpec
from cldfbench import Dataset as BaseDataset
import yaml


class KoehnPDFParser:
    def __init__(self):

        self.line_mapping = json.load(open("etc/line_mapping.json"))
        line_moves = json.load(open("etc/line_movements.json"))
        line_splits = json.load(open("etc/line_splits.json"))
        col_mapping = json.load(open("etc/col_mapping.json"))
        unit_check = json.load(open("etc/unit_check.json"))
        insertions  = json.load(open("etc/insertions.json"))
        replacements  = json.load(open("etc/replacements.json"))
        broken_units = json.load(open("etc/broken_units.json"))

        segments = pd.read_csv("etc/profile.csv")
        segment_list = [
            {"Grapheme": x, "mapping": y}
            for x, y in dict(zip(segments["Ortho"], segments["IPA"])).items()
        ]
        tokenizer = Tokenizer(Profile(*segment_list))

        texts = pd.read_csv("etc/texts.csv", index_col=0, keep_default_na=False)
        texts = texts.to_dict("index")

        obj_corr = {
            "ê": "ẽ",
            "~": "Ṽ",
        }

        gloss_corr = {
            "VOC~e": "VOCÊ",
            "NCM velho": "NCM-velho",
            "13S ser": "13S-ser",
            'V0C"e': "VOCÊ",
            "f azer": "fazer",
            "ref lx": "reflx",
            "di zer": "dizer",
        }

        def get_pages(text, text_folder, page_map):
            folder_path = os.path.join("temp", text_folder)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            for text_page, total_page in page_map.items():
                target_file = f"{text}_{text_page}.pdf"
                target_path = os.path.join("temp", text_folder, target_file)
                source_path = f"raw/single_pages/out-{str(total_page).zfill(3)}.pdf"
                os.popen(f"cp {source_path} {target_path}")

        def print_example(ex):
            for key in [
                "ID",
                "Primary_Text",
                "Analyzed_Word",
                "Gloss",
                "gramm",
                "Translated_Text",
            ]:
                if key in ex:
                    print(f"{key}:\t{ex[key]}")
            print("")

        def identify_units(lines, label, page_break=False):
            units = []
            current_unit = []
            for line in lines:
                if label in line:
                    if current_unit and (len(current_unit[0]) > 3 or page_break):
                        units.append(current_unit)
                    current_unit = []
                    page_break = False
                if line.strip() == "":
                    continue
                current_unit.append(line)
            units.append(current_unit)
            return units

        def print_partial_analysis(unit_raw, unit, keys):
            for i, l in enumerate(unit_raw):
                print(f"{i} {l}")
                if (i + 1) % len(keys) == 0:
                    print("")
            print_example(unit)

        def parse_unit(unit_raw, label, text):
            # print("\n".join(unit_raw))
            if label not in unit_raw[0]:
                return None
            nr = unit_raw.pop(0).replace(label, "").strip()
            id = f"{text}-{nr}"
            unit = {"ID": id, "Text_ID": text, "Sentence_Number": nr}
            # print(unit["ID"])
            # print(len(unit_raw))
            keys = ["Primary_Text", "Analyzed_Word", "Gloss", "gramm"]
            specific_keys = {"Translated_Text": -1}
            if id not in unit_check or unit_check[id]["initial"] != 1:
                print("Input:")
                print_partial_analysis(unit_raw, unit, keys)

            if unit["ID"] in self.line_mapping:
                target_lines = {
                    line: col_mapping[key]
                    for key, sublist in self.line_mapping[unit["ID"]].items()
                    for line in sublist
                }
                for line, key in dict(
                    sorted(target_lines.items(), key=lambda item: item[0], reverse=True)
                ).items():
                    if len(unit_raw) == 0:
                        print(f"Overparsing {id}:")
                        print_partial_analysis(unit_raw, unit, keys)
                        # raise ValueError("Empty lines")
                    if key in keys:
                        keys.remove(key)
                    if key in specific_keys:
                        del specific_keys[key]
                    if key not in unit:
                        unit[key] = unit_raw.pop(line)
                    else:
                        unit[key] = unit_raw.pop(line) + " " + unit[key]
            for key, val in specific_keys.items():
                unit[key] = unit_raw.pop(val)
            if len(keys) > 1:
                if len(unit_raw) % len(keys) == 0:
                    while len(unit_raw) != 0:
                        for key in keys:
                            if key not in unit:
                                unit[key] = unit_raw.pop(0)
                            else:
                                unit[key] += " " + unit_raw.pop(0)
                else:
                    print(f"Length mismatch in {id} ({len(unit_raw)}):")
                    print_partial_analysis(unit_raw, unit, keys)
                    # raise ValueError("Invalid number of lines in unit")
            for key in unit.keys():
                if key == "Gloss":
                    unit[key] = (
                        unit[key]
                        .replace("*", "_")
                        .replace("___", "***")
                        .replace("_ ", "_")
                        .replace("/ ", "/")
                    )
                    for orig, repl in gloss_corr.items():
                        unit[key] = unit[key].replace(orig, repl)
                unit[key] = unit[key].replace("  ", " ").replace("AAA", "***").strip()
                if key in ["Gloss", "Analyzed_Word"]:
                    unit[key] = unit[key].replace("- ", "-").replace(" -", "-")
                if key in ["Analyzed_Word", "Primary_Text"]:
                    unit[key] = unit[key].translate(
                        {ord(x): y for x, y in obj_corr.items()}
                    )
            if id not in unit_check:
                print_example(unit)
                val = input("Is parsing good? [y]es [N]o\n")
                if val != "y":
                    return None
                else:
                    if id not in unit_check:
                        unit_check[id] = {}
                    unit_check[id]["initial"] = 1
            return unit

        delim = [".", ";", ",", "!", "?", "*"]

        def ipaify(string, obj=False):
            string = string.lower()
            string = unicodedata.normalize("NFD", string)
            string = string.translate({ord(x): "" for x in delim})
            return tokenizer(string).replace(" ", "").replace("#", " ")

        parsed = []

        for text, metadata in texts.items():
            if metadata["Label"]:
                text_label = metadata["Label"]
            else:
                text_label = text
            text_folder = text + "_pages"
            page_map = {}
            for text_page, total_page in enumerate(
                range(metadata["Start"], metadata["End"] + 1)
            ):
                page_map[text_page + 1] = total_page
            if not os.path.isdir(text_folder):
                get_pages(text, text_folder, page_map)
            if text not in ["po2", "ner1"]:
                continue
            page_break = False
            for text_page, total_page in page_map.items():
                page_id = f"{text}_{text_page}"
                print(page_id)
                page_pdf = os.path.join("temp", text_folder, f"{page_id}.pdf")
                raw_text = parser.from_file(page_pdf)
                text_file = os.path.join("temp", text_folder, f"{page_id}.txt")
                lines = raw_text["content"].split("\n")
                if page_id in line_moves:
                    for move in line_moves[page_id]:
                        lines.insert(move[1] - 1, lines.pop(move[0]))
                if page_id in line_splits:
                    for lineno, indices in line_splits[page_id].items():
                        indices = [0] + indices
                        line = lines[int(lineno)]
                        split_line = [
                            line[i:j] for i, j in zip(indices, indices[1:] + [None])
                        ]
                        del lines[int(lineno)]
                        for new_line in reversed(split_line):
                            lines.insert(int(lineno), new_line)
                f = open(text_file, "w")
                for i, line in enumerate(lines):
                    f.write(f"{i} {line}\n")
                f.close()
                units = identify_units(lines, text_label, page_break)
                for u_c, unit in enumerate(units):
                    if page_id in broken_units and u_c == len(units) - 1:
                        page_break = True
                        partial_unit_1 = unit
                    else:
                        if page_break:
                            unit = partial_unit_1 + unit
                            page_break = False
                        parsed_unit = parse_unit(unit, text_label, text)
                        # print_example(parsed_unit)
                        if not parsed_unit:
                            continue
                        if page_break:
                            print(parsed_unit, partial_unit_1)
                        parsed_unit["page"] = total_page
                        unit_id = parsed_unit["ID"]
                        if parsed_unit["ID"] in insertions:
                            for key, inserts in insertions[parsed_unit["ID"]].items():
                                key = col_mapping[key]
                                for position, add_text in inserts.items():
                                    pos = int(position)
                                    print(f"""Inserting {add_text} in {key} field at position {pos} in unit {parsed_unit["ID"]}""")
                                    parsed_unit[key] = parsed_unit[key][:pos] + add_text + parsed_unit[key][pos:]
                        if unit_id in replacements:
                            for key, repl in replacements[unit_id].items():
                                for a, b in repl.items():
                                    print(f"Replacing {a} with {b} in {key} in {unit_id}")
                                    parsed_unit[col_mapping[key]] = parsed_unit[col_mapping[key]].replace(a, b)
                                    print(parsed_unit)
                        if "Primary_Text" in parsed_unit:
                            parsed_unit["pnm"] = ipaify(parsed_unit["Primary_Text"])
                            if "Analyzed_Word" in parsed_unit:
                                parsed_unit["pnm_parsed"] = ipaify(
                                    parsed_unit["Analyzed_Word"], obj=True
                                )
                                parsed.append(parsed_unit)

        df = pd.DataFrame.from_dict(parsed)
        df["Language_ID"] = "apa"
        df["Analyzed_Word"] = df["Analyzed_Word"].apply(
            lambda x: "\t".join(x.split(" "))
        )
        df["Gloss"] = df["Gloss"].apply(lambda x: "\t".join(x.split(" ")))
        df.rename(columns={"trash": "Comments", "ID": "Example_ID"}, inplace=True)
        df.index.name = "ID"
        df.to_csv(os.path.join("cldf", "examples.csv"))
        sample_list = ["ner1-008", "ner1-025"]
        df[df["Example_ID"].isin(sample_list)].to_csv(
            os.path.join("cldf", "examples_sample.csv")
        )
        with open("etc/unit_check.json", "w") as outfile:
            json.dump(unit_check, outfile, indent=2)


class Dataset(BaseDataset):
    dir = pathlib.Path(__file__).parent
    id = "textos_apalai"

    def cldf_specs(self):  # A dataset must declare all CLDF sets it creates.
        return CLDFSpec(
            dir=self.cldf_dir, module="Generic", metadata_fname="cldf-metadata.json"
        )
        return super().cldf_specs()

    def cmd_download(self, args):
        pass

        """
        Download files to the raw/ directory. You can use helpers methods of `self.raw_dir`, e.g.

        >>> self.raw_dir.download(url, fname)
        """
        # pass

    def cmd_makecldf(self, args):
        args.writer.cldf.add_component("LanguageTable")
        args.writer.cldf.add_component(
            "ExampleTable",
            "Text_ID",
            {"name": "Sentence_Number", "datatype": "integer"},
            {"name": "Phrase_Number", "datatype": "integer"},
        )
        args.writer.cldf.remove_columns("ExampleTable", "Gloss", "Analyzed_Word")
        args.writer.cldf.add_columns("ExampleTable",
             {
                "dc:description": "The sequence of words of the primary text to be aligned with glosses",
                "dc:extent": "multivalued",
                "datatype": "string",
                "propertyUrl": "http://cldf.clld.org/v1.0/terms.rdf#analyzedWord",
                "required": False,
                "separator": "\t",
                "name": "Analyzed_Word",
            },
            {
                "dc:description": "The sequence of glosses aligned with the words of the primary text",
                "dc:extent": "multivalued",
                "datatype": "string",
                "propertyUrl": "http://cldf.clld.org/v1.0/terms.rdf#gloss",
                "required": False,
                "separator": "\t",
                "name": "Gloss",
            })
        args.writer.cldf.add_table("texts.csv", "ID", "Title")
        args.writer.cldf.add_foreign_key("ExampleTable", "Text_ID", "texts.csv", "ID")

        args.writer.objects["LanguageTable"].append(
            {"ID": "apa", "Name": "Apalaí", "Glottocode": "apal1257"}
        )

        for i, row in pd.read_csv("etc/texts.csv").iterrows():
            args.writer.objects["texts.csv"].append(
                {"ID": row["ID"], "Title": row["Title"]}
            )

        p = KoehnPDFParser()

        """
        Convert the raw data to a CLDF dataset.

        >>> args.writer.objects['LanguageTable'].append(...)
        """
