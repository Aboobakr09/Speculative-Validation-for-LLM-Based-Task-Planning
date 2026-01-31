from planner.llm_client import get_llm_client
from planner.translator import ActionTranslator
from planner.contextual_baseline import ContextualBaseline
from simulator.symbolic_home import SymbolicHome
from planner.repair_first import RepairFirstPlanner

# Initialize
llm = get_llm_client()
translator = ActionTranslator()
planner = ContextualBaseline(llm, translator)
#planner = RepairFirstPlanner(llm, translator)  # Instead of Contextual
sim = SymbolicHome()

# Assign a task
task = "turn on the lamp"

# Run
print(f"Task: {task}\n")
result = planner.solve(instruction=task, simulator=sim, goal_spec=[])

# Results
print(f"\n{'='*40}")
print(f"Success: {result['success']}")
print(f"Steps executed: {result['steps_executed']}/{result['total_steps']}")
print(f"Plan: {result.get('translated_steps', [])}")
print(f"\nRaw LLM plan:\n{result.get('raw_plan', 'N/A')}")
print(f"\nExecution trace: {result.get('executed_trace', [])}")
print(f"Failure reason: {result.get('failure_reason', 'None')}")
