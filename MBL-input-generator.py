import pandas as pd
import re
import os

def prepare_input():
    try:
        input_df = pd.read_csv("lexicon.csv", header=None, names=["Word"])
    except:
        print("lexicon.csv not correctly formatted.")
        quit()
    try:
        CELEX = pd.read_csv("dpw.csv").drop(labels=["IdNum", "Inl", "IdNumLemma","PhonCVBR", "PhonSylBCLX"], axis=1).drop_duplicates()
    except:
        print("dpw.csv not correctly formatted.")
        quit()
    input_df = pd.merge(input_df, CELEX, on="Word", how="left")
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
                output_dict_stress[description] += "+"
            else:
                output_dict_stress[description] += "-"
            syllable = syllable.removeprefix("'")
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
        padding = ["-"] * (max_len_stress - len(output_dict_stress[description]))
        output_dict_stress[description] = padding + output_dict_stress[description]
    for description in output_dict_transcription:
        padding = ["="] * (max_len_trans - len(output_dict_transcription[description]))
        output_dict_transcription[description] = padding + output_dict_transcription[description]
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

def underspecification(input_df):
    last_syllable = input_df.columns[-1]
    input_df[last_syllable] = input_df[last_syllable].apply(lambda x: x[:-1] + "?" if x.endswith(("d","t")) else x)
    input_df[last_syllable] = input_df[last_syllable].apply(lambda x: x[:-1] + "€" if x.endswith(("x","G")) else x)
    input_df[last_syllable] = input_df[last_syllable].apply(lambda x: x[:-1] + "£" if x.endswith(("p","b")) else x)
    return input_df

def final_letter(input_df):
    input_df["Final_letter"] = input_df["Word"].str[-1]
    return input_df

if __name__ == "__main__":
    if not os.path.exists("dpw.csv"):
        print("Could not find dpw.csv")
    if not os.path.exists("lexicon.csv"):
        print("Could not find lexicon.csv")
    if not os.path.exists("dpw.csv") or not os.path.exists("lexicon.csv"):
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
        underspecificationsetting = input("Use underspecification for word-final obstruents (y/n): ")
        if underspecificationsetting in ("y","n"):
            break
    while True:
        finallettersetting = input("Include final letter (y/n): ")
        if finallettersetting in ("y","n"):
            break
    path_output = input("Name of output file: ").strip().strip('"').strip("'")
    if not path_output.lower().endswith(".csv"):
        path_output += ".csv"
    input_df = prepare_input()
    stress_dict, transcription_dict = parse_phonetic_transcription(input_df)
    stress_dict, transcription_dict = padding(stress_dict, transcription_dict)
    input_df = merge_output(stress_dict, transcription_dict, input_df, stress_limiter, syllable_limiter)
    if underspecificationsetting == "y":
        input_df = underspecification(input_df)
    if finallettersetting == "y":
        input_df = final_letter(input_df)
    input_df.to_csv(path_or_buf=path_output, header=False, index=False)
    print(f"Finished successfully. Output written to {path_output}")