import json

with open("artifacts/episode_stats_sac.json") as f:
    data = json.load(f)

total_eps = len(data)
laps = [d for d in data if d.get("termination_reason") == "lap_complete"]
print(f"Total episodes: {total_eps}")
print(f"Lap completions: {len(laps)} ({len(laps)/total_eps*100:.1f}%)")

tight_eps = [d for d in data if d.get("fuel_budget", 10) < 4.0]
mid_eps = [d for d in data if 4.0 <= d.get("fuel_budget", 10) <= 6.0]
loose_eps = [d for d in data if d.get("fuel_budget", 0) > 6.0]

tight_laps = [d for d in tight_eps if d.get("termination_reason") == "lap_complete"]
mid_laps = [d for d in mid_eps if d.get("termination_reason") == "lap_complete"]
loose_laps = [d for d in loose_eps if d.get("termination_reason") == "lap_complete"]

print(f"Tight budget eps (<4.0): {len(tight_eps)}, completions: {len(tight_laps)}")
print(f"Mid budget eps (4-6): {len(mid_eps)}, completions: {len(mid_laps)}")
print(f"Loose budget eps (>6): {len(loose_eps)}, completions: {len(loose_laps)}")

if laps:
    avg_lap = sum(d.get("lap_time_sec", 0) for d in laps) / len(laps)
    min_lap = min(d.get("lap_time_sec", 0) for d in laps)
    print(f"Average lap time: {avg_lap:.2f}s, Best: {min_lap:.2f}s")
