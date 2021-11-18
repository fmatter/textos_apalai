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


class ApalaiParser():
    
    def __init__(self):

        line_mapping = json.load(open("etc/line_mapping.json"))
        line_moves = json.load(open("etc/line_movements.json"))
        line_splits = json.load(open("etc/line_splits.json"))

        segments = pd.read_csv("etc/profile.csv")
        segment_list = [
            {"Grapheme": x, "mapping": y}
            for x, y in dict(zip(segments["Ortho"], segments["IPA"])).items()
        ]
        tokenizer = Tokenizer(Profile(*segment_list))

        texts = {
            "ner2": {"start_page": 2, "end_page": 15},
            "po2": {"start_page": 16, "end_page": 26},
            "ner1": {"start_page": 27, "end_page": 38, "label": "nerl"},
        }

        obj_corr = {
            "ê": "ẽ",
            "~": "Ṽ",
        }

        gloss_corr = {"NCM velho": "NCM-velho", "13S ser": "13S-ser"}

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
            for key in ["ID", "Primary_Text", "Analyzed_Word", "Gloss", "gramm", "Translated_Text"]:
                if key in ex:
                    print(f"{key}:\t{ex[key]}")
            print("")

        def identify_units(lines, label):
            units = []
            current_unit = []
            for line in lines:
                if label in line:
                    if current_unit and len(current_unit[0]) > 4:
                        units.append(current_unit)
                    current_unit = []
                if line.strip() == "":
                    continue
                current_unit.append(line)
            units.append(current_unit)
            return units

        def parse_unit(unit_raw, label, text):
            # print("\n".join(unit_raw))
            if label not in unit_raw[0]:
                return False
            nr = unit_raw.pop(0).strip(label).strip()
            unit = {"ID": f"{text}-{nr}", "Text_ID": text, "Sentence_Number": nr}
            # print(unit["ID"])
            # print(len(unit_raw))
            keys = ["Primary_Text", "Analyzed_Word", "Gloss", "gramm"]
            specific_keys = {"Translated_Text": -1}
            if unit["ID"] in line_mapping:
                target_lines = {
                    line: key
                    for key, sublist in line_mapping[unit["ID"]].items()
                    for line in sublist
                }
                for line, key in dict(
                    sorted(target_lines.items(), key=lambda item: item[0], reverse=True)
                ).items():
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
            if len(unit_raw) % 2 == 0:
                while len(unit_raw) != 0:
                    for key in keys:
                        if key not in unit:
                            unit[key] = unit_raw.pop(0)
                        else:
                            unit[key] += " " + unit_raw.pop(0)
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
            return unit

        delim = [".", ";", ",", "!", "?", "*"]

        def ipaify(string, obj=False):
            string = string.lower()
            string = unicodedata.normalize("NFD", string)
            string = string.translate({ord(x): "" for x in delim})
            return tokenizer(string).replace(" ", "").replace("#", " ")

        parsed = []

        for text, metadata in texts.items():
            if "label" in metadata:
                text_label = metadata["label"]
            else:
                text_label = text
            text_folder = text + "_pages"
            page_map = {}
            for text_page, total_page in enumerate(
                range(metadata["start_page"], metadata["end_page"] + 1)
            ):
                page_map[text_page + 1] = total_page
            if not os.path.isdir(text_folder):
                get_pages(text, text_folder, page_map)
            if text != "ner1":
                continue
            for text_page, total_page in page_map.items():
                page_id = f"{text}_{text_page}"
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
                units = identify_units(lines, text_label)
                for unit in units:
                    parsed_unit = parse_unit(unit, text_label, text)
                    parsed_unit["page"] = total_page
                    if "Primary_Text" in parsed_unit:
                        parsed_unit["pnm"] = ipaify(parsed_unit["Primary_Text"])
                        if "Analyzed_Word" in parsed_unit:
                            parsed_unit["pnm_parsed"] = ipaify(parsed_unit["Analyzed_Word"], obj=True)
                            parsed.append(parsed_unit)

        df = pd.DataFrame.from_dict(parsed)
        df["Language_ID"] = "apa"
        df.rename(columns={"trash": "Comments"})
        df.to_csv(os.path.join("cldf", "examples.csv"), index=False)
        
class Dataset(BaseDataset):
    dir = pathlib.Path(__file__).parent
    id = "cldftest"

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
        args.writer.cldf.add_table("texts.csv", "ID", "Title")
        args.writer.cldf.add_foreign_key("ExampleTable", "Text_ID", "texts.csv", "ID")

        args.writer.objects["LanguageTable"].append(
            {"ID": "apa", "Name": "Apalaí", "Glottocode": "apal1257"}
        )

        print("YES?")
        p = ApalaiParser()
        # parse_apalai()

        # for text_id, title, lines in iter_texts(self.raw_dir.read('Qiang-2.txt').split('\n')):
        #     args.writer.objects['texts.csv'].append({'ID': text_id, 'Title': title})
        #     text, gloss = [], []
        #     for igt in iter_igts(lines):
        #         text.extend(igt[1])
        #         gloss.extend(igt[2])
        #     for sid, sentence in enumerate(iter_sentences(zip(text, gloss)), start=1):
        #         for pid, phrase in enumerate(iter_phrases(sentence), start=1):
        #             example_number += 1
        #             args.writer.objects['ExampleTable'].append({
        #                 'ID': example_number,
        #                 'Primary_Text': ' '.join(p[0] for p in phrase),
        #                 'Analyzed_Word': [p[0] for p in phrase],
        #                 'Gloss': [p[1] for p in phrase],
        #                 'Text_ID': text_id,
        #                 'Language_ID': 'qiang',
        #                 'Sentence_Number': sid,
        #                 'Phrase_Number': pid,
        #             })
        """
        Convert the raw data to a CLDF dataset.

        >>> args.writer.objects['LanguageTable'].append(...)
        """
