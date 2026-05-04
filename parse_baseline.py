import json

try:
    with open('backups/artifacts_20260429_021221/episode_stats_eval.json') as f:
        stats = json.load(f)
    print("Baseline stats:")
    for s in stats:
        if s.get('termination_reason') == 'lap_complete':
            print(f"Ep: {s.get('episode')}, Fuel Used: {s.get('fuel_consumed'):.2f}, Lap Time: {s.get('lap_time_sec')}")
except Exception as e:
    print(e)
