import pandas as pd
from prefixspan import PrefixSpan

def recommend(data: pd.DataFrame, group: str, order: str, item: str, seq: list):
    """
    Extrait des séquences avec PrefixSpan à partir de données "data" identifiées
    par "group" et suivant les attributs "item" pour prédire le prochain item 
    d'une séquence partielle.
    """

    sequences = (
        data
        .sort_values([group, order])
        .reset_index(drop=True) 
        .groupby(group)[item]
        .apply(list).tolist()
    )

    ps = PrefixSpan(sequences)
    ps.minlen, ps.maxlen = 2, 5

    support_min = 0.02 # support significatif
    motifs = ps.frequent(int(support_min*len(sequences)))

    motif_df = (
        pd.DataFrame(motifs, columns=['support', 'motif'])
        .sort_values(['support'], ascending=[False])
        .reset_index(drop=True)
    )

    rules = {}
    for support, motif in motif_df.itertuples(index=False):
        for i in range(1, len(motif)):
            # tuple = hashable
            prefix = tuple(motif[:i])
            suffix = motif[i:]
            if prefix not in rules:
                rules[prefix] = (suffix, support)

    # default = most frequent
    if tuple(seq) not in rules:
        counts = data[item].value_counts()
        return ([counts.index[0]], int(counts.iloc[0]))
    return rules[tuple(seq)]
