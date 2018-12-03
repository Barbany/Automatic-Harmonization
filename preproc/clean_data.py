""" Apply pre-processing techniques to raw data """
import pandas as pd
import json
import numpy as np


def create_clean_file(path, clean_data_file, mapping_file):
    df = pd.read_csv(path + 'all_annotations.tsv', sep='\t')
    # Process data:
    #   - Keep chords without end of phrase indicator (nor key)
    #   - Remove the start and the end of the pedal from the chords
    #   - Remove global and local key as they are kept as features
    #   - Keep end of phrase boolean

    df = df[['chord', 'mov', 'global_key', 'local_key', 'phraseend']]

    # Remove end of phrase
    df['chord'] = df['chord'].str.replace('\\\\\\\\', '')

    # Remove pedal start/end [,]
    df['chord'] = df['chord'].str.replace(']', '')
    df['chord'] = df['chord'].str.split('\[').str[0]

    # Remove global/ local key - starts with dot
    keys_idx = df['chord'].str.startswith('.').index
    # If it is ".V.ii", then "V" is either local or global key and ii a chord -> keep only the chord
    # If it is ".Eb" then the global key is false (atonal) and we want only the chord
    df['chord'] = df['chord'][keys_idx].str.split('.').str[-1]

    # TODO: tests to see if we need to further reduce the size
    # Optional: - remove relative foot '/' and changes (.)
    #           - too many empty values to be kept as features
    df['chord'] = df['chord'].str.split('/').str[0]         # relative foot
    df['chord'] = df['chord'].str.replace(r"\(.*\)", "")    # changes

    print("vocabulary size", df['chord'].nunique())

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
