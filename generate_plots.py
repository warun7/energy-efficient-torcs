import json
import matplotlib.pyplot as plt
import numpy as np

with open('eval_results_conditioned.json') as f:
    data = json.load(f)

budgets = [d['budget'] for d in data]
energy_used = [d['mean_energy_used'] for d in data]
completion_rates = [d['completion_rate'] * 100 for d in data]

# Plot 1: Fuel Consumed vs Budget
filtered_budgets = [b for b in budgets if b >= 4.0]
filtered_energy = [e for b, e in zip(budgets, energy_used) if b >= 4.0]

plt.figure(figsize=(8, 5))
plt.plot(filtered_budgets, filtered_energy, marker='o', linestyle='-', color='b', linewidth=2)
# REMOVED fake baseline 5.3 line
plt.xlabel('Allocated Fuel Budget')
plt.ylabel('Mean Energy Consumed')
plt.title('Fuel Consumption vs. Allocated Budget')
plt.grid(True)
plt.savefig('artifacts/budget_consumption.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar Chart: Fuel Consumption Comparison
plt.figure(figsize=(8, 5))
completed_budgets = [str(b) for b in budgets if b >= 5.0]
completed_energy = [e for b, e in zip(budgets, energy_used) if b >= 5.0]

bars = plt.bar(completed_budgets, completed_energy, color='cornflowerblue', edgecolor='black', width=0.5)
for bar, energy in zip(bars, completed_energy):
    plt.text(bar.get_x() + bar.get_width()/2, energy + 0.05, f"{energy:.2f}", ha='center', va='bottom', fontsize=10)

plt.xlabel('Allocated Fuel Budget (Unconstrained Laps)')
plt.ylabel('Mean Energy Consumed')
plt.title('Fuel Consumption Across Varying Unconstrained Budgets')
plt.ylim(0, max(completed_energy) + 0.5)
plt.grid(axis='y', alpha=0.3)
plt.savefig('artifacts/fuel_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

