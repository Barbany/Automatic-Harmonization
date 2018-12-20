""" Apply pre-processing techniques to raw data """
import json

import numpy as np
import pandas as pd


def create_clean_file(path, clean_data_file, mapping_file, verbose=False):
    df = pd.read_csv(path + 'all_annotations.tsv', sep='\t')
    # Process data:
    #   - Keep chords without end of phrase indicator (nor key)
    #   - Remove the start and the end of the pedal from the chords
    #   - Remove global and local key as they are kept as features
    #   - Keep end of phrase boolean

    df = df[['chord', 'altchord', 'mov', 'length', 'global_key', 'local_key', 'pedal', 'form', 'figbass',
        'changes', 'relativeroot', 'phraseend']]

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

    # Remove relative foot '/' and changes (.)
    #  - too many empty values to be kept as features
    df['chord'] = df['chord'].str.split('/').str[0]         # relative foot
    df['chord'] = df['chord'].str.replace(r"\(.*\)", "")    # changes

    print("vocabulary size", df['chord'].nunique())

    # Substitute NaN values (interpreted as numerals) by string
    for col, type_ in df.dtypes.iteritems():
        if df[col].isnull().any():
            if type_ == np.object:
                df[col] = df[col].fillna('NaN')
            else:
                df[col] = df[col].fillna(-1)

    print(df.head())
    # Create mapping for all non-numerical (nor boolean) features
    mappings = {}
    for col, type_ in df.dtypes.iteritems():
        if verbose:
            print('Mapping column', col, 'from characters to floats')
        if type_ == np.object:
            mappings.update(dict({col: dict([(unique_label, float(idx))
                                              for idx, unique_label in enumerate(np.unique(df[col].values))])}))

    with open(mapping_file, 'w') as outfile:
        json.dump(mappings, outfile)

    # Save processed file and return it to avoid reloading (Don't write indices and save it as TSV - tab as separator)
    df.to_csv(clean_data_file, index=False, sep='\t')
    return df
