import pandas as pd

# Load both CSVs
df_perturbed = pd.read_csv("simulation_results_initorient.csv")
df_unperturbed = pd.read_csv("simulation_results.csv")

# Make sure both have matching keys
df_perturbed['Simulation'] = df_perturbed['Simulation'].astype(str)
df_unperturbed['Simulation'] = df_unperturbed['Simulation'].astype(str)

# Add suffixes so we can differentiate metrics later
merged = pd.merge(
    df_unperturbed,
    df_perturbed,
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

