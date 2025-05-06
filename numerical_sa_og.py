import pandas as pd

# Load both CSVs
df_afsa = pd.read_csv("simulation_results.csv")
df_afsa_sa = pd.read_csv("simulation_results_sa.csv")

# Make sure both have matching keys
df_afsa['Simulation'] = df_afsa['Simulation'].astype(str)
df_afsa_sa['Simulation'] = df_afsa_sa['Simulation'].astype(str)

# Add suffixes so we can differentiate metrics later
merged = pd.merge(
    df_afsa,
    df_afsa_sa,
    on=['Simulation', 'Robot ID'],
    suffixes=('_unpert', '_pert')
)

# Compute % change in key metrics
for metric in ['Path Length', 'Time Taken', 'Collisions']:
    merged[f'{metric}_instability'] = (
        (merged[f'{metric}_pert'] - merged[f'{metric}_unpert']) /
        merged[f'{metric}_unpert']
    ) * 100

# View the instability results
print(merged[['Simulation', 'Robot ID',
              'Path Length_instability',
              'Time Taken_instability',
              'Collisions_instability']])

