""" Apply pre-processing techniques to raw data """
import pandas as pd


def create_clean_file(path, clean_data_file):
    df = pd.read_csv(path + 'all_annotations.tsv', sep='\t')
    # Process data:
    #  - Keep chords without end of phrase indicator (nor key)
    #  - Keep end of phrase boolean
    df = df[['chord', 'phraseend']]
    idx = df['chord'][df['chord'].str.endswith('\\\\\\\\')].index
    df['chord'][idx] = df['chord'][idx].str.replace('\\\\\\\\', '')

    # Save processed file and return it to avoid reloading (Don't write indices and save it as TSV - tab as separator)
    df.to_csv(clean_data_file, index=False, sep='\t')
    return df
