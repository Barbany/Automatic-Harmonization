""" Apply pre-processing techniques to raw data """
import pandas as pd


def chords_phrases(path, clean_data_file):
    df = pd.read_csv(path + 'all_annotations.tsv', sep='\t')
    # Process data:
    #  - Keep chords without end of phrase indicator (nor key)
    #  - Keep end of phrase boolean
    df = df[['chord', 'phraseend']]
    idx = df['chord'][df['chord'].str.endswith('\\\\\\\\')].index
    df['chord'][idx] = df['chord'][idx].str.replace('\\\\\\\\', '')

    # Save processed file and return it to avoid reloading
    df.to_csv(clean_data_file, sep='\t')
    return df
