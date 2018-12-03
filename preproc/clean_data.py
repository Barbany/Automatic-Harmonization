""" Apply pre-processing techniques to raw data """
import pandas as pd
import json
import numpy as np


def create_clean_file(path, clean_data_file, mapping_file):
    df = pd.read_csv(path + 'all_annotations.tsv', sep='\t')
    # Process data:
    #  - Keep chords without end of phrase indicator (nor key)
    #  - Keep end of phrase boolean
    df = df[['chord', 'phraseend']]
    idx = df['chord'][df['chord'].str.endswith('\\\\\\\\')].index
    df['chord'][idx] = df['chord'][idx].str.replace('\\\\\\\\', '')

    # Create mapping for all non-numerical (nor boolean) features
    mappings = []
    for col, type_ in df.dtypes.iteritems():
        if type_ == np.object:
            vocabulary_size = np.unique(df[col].values)
            mappings.append({col:
                    dict([(unique_label, float(idx)) for idx, unique_label in enumerate(np.unique(df[col].values))])
                })

    with open(mapping_file, 'w') as outfile:
        json.dump(mappings, outfile)

    # Save processed file and return it to avoid reloading (Don't write indices and save it as TSV - tab as separator)
    df.to_csv(clean_data_file, index=False, sep='\t')
    return df
