import json

with open('artifacts/episode_stats_sac.json') as f:
    stats = json.load(f)

completed = [s for s in stats if s['termination_reason'] == 'lap_complete']
print(f"Total completed laps: {len(completed)}")
for s in completed[:10]:
    print(f"Ep: {s['episode']}, Budget: {s['fuel_budget']:.2f}, Fuel Used: {s['fuel_consumed']:.2f}, Lap Time: {s['lap_time_sec']}")

print("...")
for s in completed[-10:]:
    print(f"Ep: {s['episode']}, Budget: {s['fuel_budget']:.2f}, Fuel Used: {s['fuel_consumed']:.2f}, Lap Time: {s['lap_time_sec']}")
