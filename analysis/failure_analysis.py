"""
Failure Mode Analysis for Language Planner Evaluation.

Analyzes where each planner failed and why RepairFirst
underperformed Contextual on some tasks.
"""

import json
import glob
import os


def load_latest_results():
    """Load the most recent results file."""
    files = glob.glob('results/raw_results_*.json')
    if not files:
        raise FileNotFoundError("No results files found in results/")
    latest = max(files, key=os.path.getctime)
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f)


def analyze_failures(data):
    """Analyze failure patterns across planners."""
    
    # Group by planner
    by_planner = {}
    for r in data:
        planner = r['planner']
        if planner not in by_planner:
            by_planner[planner] = []
        by_planner[planner].append(r)
    
    print("=" * 70)
    print("FAILURE MODE ANALYSIS")
    print("=" * 70)
    
    # Overall stats
    print("\n## Overall Success Rates")
    for planner, results in by_planner.items():
        successes = sum(1 for r in results if r['success'])
        print(f"  {planner}: {successes}/{len(results)} ({100*successes/len(results):.0f}%)")
    
    # Find cases where RepairFirst failed but Contextual succeeded
    print("\n" + "=" * 70)
    print("## RepairFirst Failed, Contextual Succeeded")
    print("=" * 70)
    
    repair_results = {r['task_id']: r for r in by_planner.get('repair_first', [])}
    contextual_results = {r['task_id']: r for r in by_planner.get('contextual', [])}
    
    for task_id, rf_result in repair_results.items():
        ctx_result = contextual_results.get(task_id)
        if not rf_result['success'] and ctx_result and ctx_result['success']:
            print(f"\n### {task_id}: {rf_result['instruction']}")
            print(f"Difficulty: {rf_result['difficulty']}")
            print(f"\nContextual (✓ succeeded):")
            print(f"  Plan: {ctx_result['translated_steps']}")
            print(f"\nRepairFirst (✗ failed):")
            print(f"  Original: {rf_result.get('original_steps', rf_result['translated_steps'])}")
            print(f"  Final: {rf_result.get('final_steps', rf_result['translated_steps'])}")
            print(f"  Repairs: {len(rf_result.get('repair_history', []))}")
            if rf_result.get('repair_history'):
                for rep in rf_result['repair_history']:
                    print(f"    - Step {rep['step_index']}: {rep['original_step']} → {rep['repaired_action']}")
            print(f"  Failure: {rf_result.get('failure_reason', 'Goal not achieved')}")
    
    # Find cases where Contextual failed but RepairFirst succeeded
    print("\n" + "=" * 70)
    print("## Contextual Failed, RepairFirst Succeeded")
    print("=" * 70)
    
    found = False
    for task_id, ctx_result in contextual_results.items():
        rf_result = repair_results.get(task_id)
        if not ctx_result['success'] and rf_result and rf_result['success']:
            found = True
            print(f"\n### {task_id}: {ctx_result['instruction']}")
            print(f"Contextual failed: {ctx_result.get('failure_reason', 'Goal not achieved')}")
            print(f"RepairFirst succeeded with {rf_result['api_calls']} API calls")
    
    if not found:
        print("\n(None found - Contextual succeeded everywhere RepairFirst did)")
    
    # Repair mechanism usage
    print("\n" + "=" * 70)
    print("## Repair Mechanism Usage")
    print("=" * 70)
    
    repairs_used = 0
    for r in by_planner.get('repair_first', []):
        n_repairs = len(r.get('repair_history', []))
        if n_repairs > 0:
            repairs_used += 1
            print(f"\n{r['task_id']}: {n_repairs} repair(s), API calls: {r['api_calls']}")
            for rep in r['repair_history']:
                print(f"  Step {rep['step_index']}: '{rep['original_step']}' → '{rep['repaired_action']}'")
                print(f"    Error: {rep['error']}")
    
    if repairs_used == 0:
        print("\n(No repairs were triggered)")
    else:
        print(f"\nTotal tasks with repairs: {repairs_used}/{len(by_planner.get('repair_first', []))}")
    
    # Both planners failed
    print("\n" + "=" * 70)
    print("## Both Contextual and RepairFirst Failed")
    print("=" * 70)
    
    for task_id, ctx_result in contextual_results.items():
        rf_result = repair_results.get(task_id)
        if not ctx_result['success'] and rf_result and not rf_result['success']:
            print(f"\n### {task_id}: {ctx_result['instruction']}")
            print(f"  Contextual: {ctx_result.get('failure_reason', 'Goal not achieved')}")
            print(f"  RepairFirst: {rf_result.get('failure_reason', 'Goal not achieved')}")
    
    return by_planner


if __name__ == "__main__":
    data = load_latest_results()
    analyze_failures(data)
