import pandas as pd

def get_probs_table(counts, sort_values: bool = False):
    
    df = pd.DataFrame(columns=['bitstring', 'probability'])

    for bitstring in counts.get_sequential_data():
        df.loc[len(df)] = [bitstring, counts.probability(bitstring)]
        
    if sort_values:
        df = df.sort_values("probability", ascending = False)
        
    return df