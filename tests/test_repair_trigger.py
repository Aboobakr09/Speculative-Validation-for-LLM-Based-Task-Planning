"""
Test that forces precondition failure to trigger repair mechanism.

Setup: Agent already holding keys, tries to pickup cup
Expected: Contextual fails, RepairFirst repairs by dropping keys first
"""

import copy
from planner.llm_client import get_llm_client
from planner.repair_first import RepairFirstPlanner
from planner.contextual_baseline import ContextualBaseline
from simulator.symbolic_home import SymbolicHome


def test_repair_trigger():
    """
    Test that RepairFirst repairs failed steps.
    
    Strategy: Use Huang baseline (NO context) which WILL make mistakes,
    then show RepairFirst uses context + repair to succeed.
    """
    
    from simulator.symbolic_home import DEFAULT_STATE
    from planner.huang_baseline import HuangBaseline
    import copy as copy_module
    
    # Standard state - agent in kitchen, cup in kitchen
    sim = SymbolicHome()
    
    print("=" * 70)
    print("REPAIR MECHANISM TEST: Huang vs RepairFirst")
    print("=" * 70)
    print(f"\nInitial: {sim.get_state_description()}")
    
    # A task that requires navigating correctly
    task = "wash hands with soap"
    goal = ["hands_washed"]
    
    print(f"\nTask: {task}")
    print("Goal:", goal)
    print("\nHuang has NO context → will likely fail on preconditions")
    print("RepairFirst has context + can repair → should succeed")
    print("-" * 70)
    
    llm = get_llm_client()
    
    # =========================================================================
    # Test Huang Baseline (NO context - will make mistakes)
    # =========================================================================
    print("\n[HUANG BASELINE - No context]")
    sim_huang = SymbolicHome()
    huang = HuangBaseline(llm)
    result_huang = huang.solve(task, sim_huang, goal)
    
    print(f"Raw plan:\n{result_huang['raw_plan']}")
    print(f"\nTranslated: {result_huang['translated_steps']}")
    print(f"\nExecution trace:")
    for step in result_huang['executed_trace']:
        status = "✓" if step['success'] else "✗"
        print(f"  {status} {step['action']} → {step.get('error') or 'OK'}")
    
    print(f"\n→ Success: {result_huang['success']}")
    print(f"→ Error: {result_huang['failure_reason']}")
    print(f"→ API calls: {result_huang['api_calls']}")
    
    # =========================================================================
    # Test Contextual (with context)
    # =========================================================================
    print("\n" + "-" * 70)
    print("\n[CONTEXTUAL BASELINE - With context]")
    sim_ctx = SymbolicHome()
    contextual = ContextualBaseline(llm)
    result_ctx = contextual.solve(task, sim_ctx, goal)
    
    print(f"Raw plan:\n{result_ctx['raw_plan']}")
    print(f"\nTranslated: {result_ctx['translated_steps']}")
    print(f"\nExecution trace:")
    for step in result_ctx['executed_trace']:
        status = "✓" if step['success'] else "✗"
        print(f"  {status} {step['action']} → {step.get('error') or 'OK'}")
    
    print(f"\n→ Success: {result_ctx['success']}")
    print(f"→ API calls: {result_ctx['api_calls']}")
    
    # =========================================================================
    # Test RepairFirst (with context + repair)
    # =========================================================================
    print("\n" + "-" * 70)
    print("\n[REPAIR-FIRST PLANNER - Context + Repair]")
    sim_repair = SymbolicHome()
    planner = RepairFirstPlanner(llm)
    result_rf = planner.solve(task, sim_repair, goal, max_repairs=3)
    
    print(f"Raw plan:\n{result_rf['raw_plan']}")
    print(f"\nOriginal steps: {result_rf['original_steps']}")
    print(f"Final steps: {result_rf['final_steps']}")
    
    print(f"\nRepair history ({len(result_rf['repair_history'])} repairs):")
    if result_rf['repair_history']:
        for r in result_rf['repair_history']:
            print(f"  Attempt {r['attempt']}, Step {r['step_index']}:")
            print(f"    Failed: '{r['original_step']}' → {r['error']}")
            print(f"    Repaired to: '{r['repaired_action']}'")
    else:
        print("  (no repairs needed)")
    
    print(f"\nExecution trace:")
    for step in result_rf['executed_trace']:
        status = "✓" if step['success'] else "✗"
        print(f"  {status} {step['action']} → {step.get('error') or 'OK'}")
    
    print(f"\n→ Success: {result_rf['success']}")
    print(f"→ API calls: {result_rf['api_calls']}")
    print(f"→ Validation attempts: {result_rf['validation_attempts']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Huang':>12} {'Contextual':>12} {'RepairFirst':>12}")
    print("-" * 65)
    print(f"{'Success':<25} {str(result_huang['success']):>12} {str(result_ctx['success']):>12} {str(result_rf['success']):>12}")
    print(f"{'API Calls':<25} {result_huang['api_calls']:>12} {result_ctx['api_calls']:>12} {result_rf['api_calls']:>12}")
    print(f"{'Repairs':<25} {'N/A':>12} {'N/A':>12} {len(result_rf['repair_history']):>12}")
    
    return result_huang, result_ctx, result_rf


if __name__ == "__main__":
    test_repair_trigger()
