import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation_results_sa_peturb.csv")
df['N'] = df['Simulation'].str.extract(r'N=(\d+)').astype(int)

grouped = df.groupby('N').agg({
    'Path Length': 'mean',
    'Time Taken': 'mean',
    'Collisions': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(grouped['N'], grouped['Path Length'], marker='o', label='Path Length')
plt.plot(grouped['N'], grouped['Time Taken'], marker='s', label='Time Taken')
plt.plot(grouped['N'], grouped['Collisions'], marker='^', label='Collisions')
plt.xlabel('Number of Robots (N)')
plt.ylabel('Average Value')
plt.title('Average Metrics vs Number of Robots')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('results/lineplot_mean_vs_n.png')

