import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation_results_sa_peturb.csv")
df['N'] = df['Simulation'].str.extract(r'N=(\d+)').astype(int)

success_rates = df.groupby('N')['Goal Reached'].mean().reset_index()
success_rates['Goal Reached'] *= 100  # Convert to %

plt.figure(figsize=(8, 6))
plt.bar(success_rates['N'], success_rates['Goal Reached'], color='skyblue')
plt.ylim(0, 100)
plt.ylabel('Success Rate (%)')
plt.xlabel('Number of Robots (N)')
plt.title('Goal Reached Success Rate by N')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('results/success_rate_by_n.png')

