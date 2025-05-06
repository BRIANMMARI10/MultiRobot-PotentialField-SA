import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("simulation_results_sa_peturb.csv")
df['N'] = df['Simulation'].str.extract(r'N=(\d+)').astype(int)

metrics = ['Steps Taken', 'Path Length', 'Time Taken', 'Collisions']
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='N', y=metric)
    plt.title(f'{metric} Distribution by N')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/boxplot_{metric.replace(" ", "_").lower()}.png')

