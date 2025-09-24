import pandas as pd
import re
import os

def prepare_input():
    try:
        CELEX_phon = pd.read_csv("dpw.csv").drop(labels=["IdNum", "Inl","PhonCVBR", "PhonSylBCLX"], axis=1).rename(columns={"Word":"Lemma", "IdNumLemma":"IdNum"})
    except Exception:
        print("dpw.csv not correctly formatted.")
        quit()
    try:
        CELEX_synt = pd.read_csv("dsl.csv").drop(labels=["Head","Inl","GendNum", "DeHetNum","PropNum","AuxNum","SubClassVNum","SubCatNum","AdvNum","CardOrdNum","SubClassPNum"], axis=1)
    except Exception:
        print("dsl.csv not correctly formatted.")
        quit()
    try:
        CELEX_morph = pd.read_csv("dml.csv", low_memory=False).rename(columns={"Head":"Lemma"})
        CELEX_morph = CELEX_morph[["Lemma","MorphStatus"]]
    except Exception:
        print("dml.csv not correctly formatted.")
        quit()
    try:
        df_singulars = pd.read_csv("lexicon_singulars.csv").drop(labels=["Frequency"], axis=1)
    except Exception:
        print("lexicon_singulars.csv not correctly formatted.")
        quit()
    try:
        df_plurals = pd.read_csv("lexicon_plurals.csv").drop_duplicates()
    except Exception:
        print("lexicon_plurals.csv not correctly formatted.")
        quit()
    CELEX_phon_wordclass = pd.merge(CELEX_phon, CELEX_synt, on="IdNum", how="inner")
    CELEX_phon_nouns = CELEX_phon_wordclass[CELEX_phon_wordclass["ClassNum"] == 1].drop(labels=["IdNum","ClassNum"], axis=1)
    CELEX_phon_nouns["Lemma"] = CELEX_phon_nouns["Lemma"].str.lower()
    CELEX_phon_nouns.drop_duplicates(inplace=True)
    singulars_plurals_df = pd.merge(df_plurals, df_singulars, on="Lemma", how="inner")
    input_df = pd.merge(singulars_plurals_df, CELEX_phon_nouns, on="Lemma", how="inner")
    input_df = pd.merge(input_df, CELEX_morph, on="Lemma", how="inner")
    input_df = input_df[input_df["MorphStatus"] == "M"]
    input_df = input_df.drop(labels=["MorphStatus"], axis=1)
    input_df["Word"] = input_df["Word"].str.lower()
    input_df.drop_duplicates(inplace=True)
    return input_df

def parse_phonetic_transcription(input_df):
    DISC_vowels = ["i","!","a","u","y","(",")","*","<","e","|","o","K","L","M","I","E","{","A","O","}","@"]
    pattern = re.compile(f"({'|'.join(re.escape(v) for v in DISC_vowels)})")
    output_dict_stress = {}
    output_dict_transcription = {}
    for description in input_df["PhonStrsDISC"]:
        description = str(description)
        output_dict_stress[description] = []
        output_dict_transcription[description] = []
        syllables = description.split("-")
        for syllable in syllables:
            if "'" in syllable:
                output_dict_stress[description].append("+")
            else:
                output_dict_stress[description].append("-")
            if syllable.startswith("'"):
                syllable = syllable[1:]
            parts = pattern.split(syllable)
            parts = [p for p in parts if p]
            if parts[0] in DISC_vowels:
                parts.insert(0, "=")
            if parts[-1] in DISC_vowels:
                parts.append("=")
            output_dict_transcription[description] += parts
    return output_dict_stress, output_dict_transcription

def padding (output_dict_stress, output_dict_transcription):
    max_len_stress = max(len(v) for v in output_dict_stress.values())
    max_len_trans = max(len(v) for v in output_dict_transcription.values())
    for description in output_dict_stress:
        pad_stress = ["-"] * (max_len_stress - len(output_dict_stress[description]))
        output_dict_stress[description] = pad_stress + output_dict_stress[description]
    for description in output_dict_transcription:
        pad_trans = ["="] * (max_len_trans - len(output_dict_transcription[description]))
        output_dict_transcription[description] = pad_trans + output_dict_transcription[description]
    return output_dict_stress, output_dict_transcription

def merge_output(output_dict_stress, output_dict_transcription, input_df, stress_limiter, syllable_limiter):
    df_stress = pd.DataFrame.from_dict(output_dict_stress, orient="index")
    df_transcription = pd.DataFrame.from_dict(output_dict_transcription, orient="index")
    if stress_limiter.upper() != "ALL" and len(df_stress.columns) > int(stress_limiter):
        df_stress = df_stress.drop(columns=df_stress.columns[:len(df_stress.columns)-int(stress_limiter)])
    if syllable_limiter.upper() != "ALL" and len(df_transcription.columns)/3 > int(syllable_limiter):
        df_transcription = df_transcription.drop(columns=df_transcription.columns[:len(df_transcription.columns)-(3*int(syllable_limiter))])
    stress_transcriptions_df = pd.merge(df_stress, df_transcription, left_index=True, right_index=True, how="left").reset_index().rename(columns={"index":"PhonStrsDISC"})
    input_df = pd.merge(input_df, stress_transcriptions_df, on="PhonStrsDISC", how="right")
    return input_df

def remove_variable_pronunciation(input_df):
    variable_pronunciations = input_df.groupby("Lemma").filter(lambda g: g["PhonStrsDISC"].nunique() > 1).reset_index(drop=True)
    input_df = input_df.groupby("Lemma").filter(lambda g: g["PhonStrsDISC"].nunique() == 1).reset_index(drop=True)
    variable_pronunciations = variable_pronunciations.drop_duplicates(subset=["Lemma"], keep="first").reset_index(drop=True)
    input_df = pd.concat([input_df, variable_pronunciations], axis=0)
    return input_df

def underspecification(input_df):
    last_syllable = input_df.columns[-1]
    input_df[last_syllable] = input_df[last_syllable].apply(lambda x: x[:-1] + "?" if x.endswith(("d","t")) else x)
    input_df[last_syllable] = input_df[last_syllable].apply(lambda x: x[:-1] + "€" if x.endswith(("x","G")) else x)
    input_df[last_syllable] = input_df[last_syllable].apply(lambda x: x[:-1] + "£" if x.endswith(("p","b")) else x)
    return input_df

def final_letter(input_df):
    input_df["Final_letter"] = input_df["Lemma"].str[-1]
    return input_df

def plural_finder(input_df, irr_plural_checker, writeirrplurals, keepirregulars):
    input_df["Plural"] = input_df["Word"].apply(lambda x: "S" if x.endswith("s") else ("EN" if x.endswith(("en","n")) else "IRR"))
    if irr_plural_checker == "y":
        for idx, row in input_df.iterrows():
            if row["Plural"] == "IRR":
                while True:
                    keepbool = input(f"Keep '{row["Word"]}' as a plural of '{row["Lemma"]}' (y/n): ").lower().strip()
                    if keepbool in ("y","n"):
                        break
                if keepbool == "n":
                    input_df.at[idx, "Plural"] = "Remove"
        deleted = input_df[input_df["Plural"] == "Remove"].iloc[:,:1]
        deleted.to_csv(path_or_buf="deleted_plurals.csv", mode="a", index=False, header=False)
        input_df = input_df[input_df["Plural"] != "Remove"]
        for lemma, group in input_df.groupby("Lemma"):
            if len(group) == 1 and group["Plural"].iloc[0] == "IRR":
                idx = group.index[0]
                word = group["Word"].iloc[0]
                input_df.at[idx, "Plural"] = "S" if word.endswith("s") else "EN"
    if writeirrplurals == "y":
        irregular_plurals = input_df[input_df["Plural"] == "IRR"]
        irregular_plurals = irregular_plurals.iloc[:,:1]
        irregular_plurals.to_csv(path_or_buf="irregular_plurals.csv", index=False, header=False)       
    if keepirregulars == "n":
        input_df = input_df.drop(input_df[input_df.Plural == "IRR"].index)
    return input_df

def var_plural_finder(input_df, var_plural_checker, writevarplurals, keepvariables):
   plural_groups = input_df.groupby("Lemma")["Plural"].agg(lambda x: set(x))
   var_lemmas = plural_groups[plural_groups.apply(lambda x: "S" in x and "EN" in x)].index
   input_df.loc[input_df["Lemma"].isin(var_lemmas), "Plural"] = "VAR"
   if var_plural_checker == "y":
        for idx, row in input_df.iterrows():
            if row["Plural"] == "VAR":
                while True:
                    keepbool = input(f"Keep '{row["Word"]}' as a plural of '{row["Lemma"]}' (y/n): ").lower().strip()
                    if keepbool in ("y","n"):
                        break
                if keepbool == "n":
                    input_df.at[idx, "Plural"] = "Remove"
        deleted = input_df[input_df["Plural"] == "Remove"].iloc[:,:1]
        deleted.to_csv(path_or_buf="deleted_plurals.csv", mode="a", index=False, header=False)
        input_df = input_df[input_df["Plural"] != "Remove"]
        for lemma, group in input_df.groupby("Lemma"):
            if len(group) == 1 and group["Plural"].iloc[0] == "VAR":
                idx = group.index[0]
                word = group["Word"].iloc[0]
                input_df.at[idx, "Plural"] = "S" if word.endswith("s") else "EN"
   input_df = input_df.drop(labels=["Word"], axis=1).drop_duplicates()
   if writevarplurals == "y":
        variable_plurals = input_df[input_df["Plural"] == "VAR"]
        variable_plurals = variable_plurals.iloc[:,:1]
        variable_plurals.to_csv(path_or_buf="variable_plurals.csv", index=False, header=False)
   if keepvariables == "n":
        input_df = input_df.drop(input_df[input_df.Plural == "VAR"].index)
   return input_df

if __name__ == "__main__":
    if not os.path.exists("dpw.csv"):
        print("Could not find dpw.csv")
    if not os.path.exists("dsl.csv"):
        print("Could not find dsl.csv")
    if not os.path.exists("dml.csv"):
        print("Could not find dml.csv")
    if not os.path.exists("lexicon_singulars.csv"):
        print("Could not find lexicon_singulars.csv")
    if not os.path.exists("lexicon_plurals.csv"):
        print("Could not find lexicon_plurals.csv")
    if not os.path.exists("dpw.csv") or not os.path.exists("dsl.csv") or not os.path.exists("dml.csv") or not os.path.exists("lexicon_singulars.csv") or not os.path.exists("lexicon_plurals.csv"):
        quit()
    while True:
        stress_limiter = input("Number of syllables in stress analysis (number/ALL): ")
        if stress_limiter.isdigit() or stress_limiter.upper() == "ALL":
            break
    while True:
        syllable_limiter = input("Number of syllables in transcription (number/ALL): ")
        if syllable_limiter.isdigit() or syllable_limiter.upper() == "ALL":
            break
    while True:
        variablepronunciationsetting = input("Collapse variable pronunciations (y/n): ").lower()
        if variablepronunciationsetting in ("y","n"):
            break
    while True:
        underspecificationsetting = input("Use underspecification for word-final obstruents (y/n): ").lower()
        if underspecificationsetting in ("y","n"):
            break
    while True:
        finallettersetting = input("Include final letter (y/n): ").lower()
        if finallettersetting in ("y","n"):
            break
    while True:
        irr_plural_checker = input("Manually check irregular plurals (y/n): ").lower()
        if irr_plural_checker in ("y","n"):
            break
    while True:
        keepirregulars = input("Keep irregular plurals in output file (y/n): ").lower()
        if keepirregulars in ("y","n"):
            break
    while True:
        writeirrplurals = input("Write irregular plurals to separate file (y/n): ").lower()
        if writeirrplurals in ("y","n"):
            break
    while True:
        var_plural_checker = input("Manually check variable plurals (y/n): ").lower()
        if var_plural_checker in ("y","n"):
            break
    while True:
        keepvariables = input("Keep variable plurals in output file (y/n): ").lower()
        if keepvariables in ("y","n"):
            break
    while True:
        writevarplurals = input("Write variable plurals to separate file (y/n): ").lower()
        if writevarplurals in ("y","n"):
            break
    path_output = input("Name of output file: ").strip().strip('"').strip("'")
    if not path_output.lower().endswith(".csv"):
        path_output += ".csv"
    input_df = prepare_input()
    stress_dict, transcription_dict = parse_phonetic_transcription(input_df)
    stress_dict, transcription_dict = padding(stress_dict, transcription_dict)
    input_df = merge_output(stress_dict, transcription_dict, input_df, stress_limiter, syllable_limiter)
    if variablepronunciationsetting == "y":
        input_df = remove_variable_pronunciation(input_df)
    if underspecificationsetting == "y":
        input_df = underspecification(input_df)
    if finallettersetting == "y":
        input_df = final_letter(input_df)
    input_df = plural_finder(input_df, irr_plural_checker, writeirrplurals, keepirregulars)
    input_df = var_plural_finder(input_df, var_plural_checker, writevarplurals, keepvariables)
    input_df.to_csv(path_or_buf=path_output, header=False, index=False)
    print(f"Finished successfully. Output written to {path_output}")