"""
Quick test: Compare Contextual vs RepairFirst on a hard task.
"""

from planner.llm_client import get_llm_client
from planner.translator import ActionTranslator
from planner.contextual_baseline import ContextualBaseline
from planner.repair_first import RepairFirstPlanner
from simulator.symbolic_home import SymbolicHome


def test_hard_task(instruction: str, goal: list):
    """Run both planners on the same hard task."""
    
    llm = get_llm_client()
    translator = ActionTranslator()
    
    print("=" * 70)
    print(f"HARD TASK TEST: {instruction}")
    print(f"Goal: {goal}")
    print("=" * 70)
    
    # Test Contextual Baseline
    print("\n" + "-" * 35)
    print("CONTEXTUAL BASELINE")
    print("-" * 35)
    
    contextual = ContextualBaseline(llm, translator)
    sim1 = SymbolicHome()
    print(f"Initial: {sim1.get_state_description()}\n")
    
    result1 = contextual.solve(instruction, sim1, goal_spec=goal)
    
    print(f"Raw Plan:\n{result1['raw_plan']}")
    print(f"\nTranslated: {result1['translated_steps']}")
    print(f"\nExecution:")
    for step in result1['executed_trace']:
        status = "✓" if step['success'] else "✗"
        err = step.get('error') or 'OK'
        print(f"  {status} {step['action']} → {err}")
    
    print(f"\n→ Success: {result1['success']}")
    print(f"→ Goal achieved: {result1['goal_achieved']}")
    print(f"→ API calls: {result1['api_calls']}")
    if result1['failure_reason']:
        print(f"→ Failure: {result1['failure_reason']}")
    
    # Test RepairFirst
    print("\n" + "-" * 35)
    print("REPAIR-FIRST PLANNER")
    print("-" * 35)
    
    repair = RepairFirstPlanner(llm, translator)
    sim2 = SymbolicHome()
    print(f"Initial: {sim2.get_state_description()}\n")
    
    result2 = repair.solve(instruction, sim2, goal_spec=goal, max_repairs=2)
    
    print(f"Raw Plan:\n{result2['raw_plan']}")
    print(f"\nOriginal steps: {result2['original_steps']}")
    print(f"Final steps: {result2['final_steps']}")
    
    if result2['repair_history']:
        print(f"\nRepair History:")
        for r in result2['repair_history']:
            print(f"  Attempt {r['attempt']}, Step {r['step_index']}:")
            print(f"    Original: {r['original_step']}")
            print(f"    Error: {r['error']}")
            print(f"    Repaired: {r['repaired_action']}")
    
    print(f"\nExecution:")
    for step in result2['executed_trace']:
        status = "✓" if step['success'] else "✗"
        err = step.get('error') or 'OK'
        print(f"  {status} {step['action']} → {err}")
    
    print(f"\n→ Success: {result2['success']}")
    print(f"→ Goal achieved: {result2['goal_achieved']}")
    print(f"→ API calls: {result2['api_calls']}")
    print(f"→ Validation attempts: {result2['validation_attempts']}")
    if result2['failure_reason']:
        print(f"→ Failure: {result2['failure_reason']}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Contextual':>15} {'RepairFirst':>15}")
    print("-" * 55)
    print(f"{'Success':<25} {str(result1['success']):>15} {str(result2['success']):>15}")
    print(f"{'Goal Achieved':<25} {str(result1['goal_achieved']):>15} {str(result2['goal_achieved']):>15}")
    print(f"{'Steps Executed':<25} {result1['steps_executed']:>15} {result2['steps_executed']:>15}")
    print(f"{'API Calls':<25} {result1['api_calls']:>15} {result2['api_calls']:>15}")


if __name__ == "__main__":
    # Test on multiple hard tasks
    test_cases = [
        # Task that requires object relocation
        ("bring the phone to the bedroom", ["phone_in_bedroom"]),
        # Task that requires multiple room visits  
        ("turn on the bedroom lamp and then wash hands", ["lamp_on", "hands_washed"]),
    ]
    
    for instruction, goal in test_cases:
        test_hard_task(instruction, goal)
        print("\n" + "=" * 70 + "\n")
