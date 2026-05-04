import json

with open('artifacts/episode_stats_sac.json') as f:
    stats = json.load(f)

print(f"Total episodes in stats: {len(stats)}")
for s in stats[:5]:
    print(s)
print("...")
for s in stats[-5:]:
    print(s)

with open('eval_results_conditioned.json') as f:
    eval_stats = json.load(f)
print("\nEval stats:")
for e in eval_stats:
    print(e)
