"""
Repair-First Speculative Planner (Novel Method).

The main contribution: Speculative validation with targeted single-step repair.

Key innovations:
1. Generate plan WITH state context
2. Validate on CLONED simulator (speculative execution)
3. On failure: Repair ONLY the failed step (not entire plan)
4. Prefix replay: Continue validation from repaired step
5. Execute validated plan on real simulator

Target metrics: Match full-regeneration accuracy with <2x Huang latency.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from planner.translator import ActionTranslator


class RepairFirstPlanner:
    """
    Repair-First Speculative Validation Planner.
    
    Algorithm:
    1. Generate initial plan with context
    2. For each step, validate on cloned simulator
    3. On first failure, call LLM to repair ONLY that step
    4. Replace step and restart validation
    5. Max repairs limit prevents infinite loops
    6. Execute validated plan on real simulator
    
    Usage:
        llm = get_llm_client()
        planner = RepairFirstPlanner(llm)
        result = planner.solve("wash hands", SymbolicHome(), max_repairs=2)
    """
    
    def __init__(self, llm_client, translator: Optional[ActionTranslator] = None):
        """
        Initialize RepairFirst planner.
        
        Args:
            llm_client: LLM client with generate() method
            translator: ActionTranslator instance
        """
        self.llm = llm_client
        self.translator = translator or ActionTranslator()
    
    def solve(
        self, 
        instruction: str, 
        simulator, 
        goal_spec: Optional[List[str]] = None,
        max_repairs: int = 2
    ) -> Dict[str, Any]:
        """
        Execute Repair-First planning method.
        
        Args:
            instruction: Natural language task description
            simulator: SymbolicHome instance
            goal_spec: Optional list of goal predicates
            max_repairs: Maximum repair attempts (default 2)
        
        Returns:
            Dict with results and detailed tracking
        """
        simulator.reset()
        api_calls = 0
        repair_history = []
        
        # =====================================================================
        # STAGE 1: Generate initial plan WITH CONTEXT
        # =====================================================================
        prompt = self._build_contextual_prompt(instruction, simulator)
        raw_plan = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        api_calls += 1
        
        # Parse and translate
        nl_steps = self._parse_steps(raw_plan)
        steps = []
        translation_failures = []
        
        for i, nl_step in enumerate(nl_steps):
            action, conf, method = self.translator.translate(nl_step)
            if action:
                steps.append(action)
            else:
                translation_failures.append({
                    "step_index": i,
                    "nl_step": nl_step
                })
        
        original_steps = steps.copy()
        
        # =====================================================================
        # STAGE 2: Speculative Validation with Repair
        # =====================================================================
        validation_attempts = 0
        final_validation_success = False
        
        for attempt in range(max_repairs + 1):
            validation_attempts += 1
            clone = simulator.clone()
            validation_passed = True
            
            for i, step in enumerate(steps):
                valid, err = clone.is_valid(step)
                
                if valid:
                    clone.execute(step)
                    continue
                
                # FAILURE at step i
                validation_passed = False
                
                if attempt < max_repairs:
                    # Try to repair this step
                    repair_prompt = self._build_repair_prompt(
                        failed_step=step,
                        error_msg=err,
                        simulator_state=clone,
                        prefix_steps=steps[:i],
                        original_instruction=instruction
                    )
                    
                    repaired_nl = self.llm.generate(repair_prompt, max_tokens=50, temperature=0.1)
                    api_calls += 1
                    
                    # Translate repaired step
                    repaired_action, conf, method = self.translator.translate(repaired_nl.strip())
                    
                    repair_history.append({
                        "attempt": attempt,
                        "step_index": i,
                        "original_step": step,
                        "error": err,
                        "repair_response": repaired_nl.strip(),
                        "repaired_action": repaired_action,
                        "success": repaired_action is not None
                    })
                    
                    if repaired_action:
                        steps[i] = repaired_action
                        break  # Restart validation from beginning
                    else:
                        # Repair failed to translate, give up on this attempt
                        break
                else:
                    # No more repair attempts allowed
                    repair_history.append({
                        "attempt": attempt,
                        "step_index": i,
                        "original_step": step,
                        "error": err,
                        "repair_response": None,
                        "repaired_action": None,
                        "success": False
                    })
                    break
            
            if validation_passed:
                final_validation_success = True
                break
        
        # =====================================================================
        # STAGE 3: Execute Validated Plan on REAL Simulator
        # =====================================================================
        executed = []
        failure_reason = None
        error_step = None
        error_type = None
        
        for i, step in enumerate(steps):
            success, err = simulator.execute(step)
            
            executed.append({
                "step_index": i,
                "action": step,
                "success": success,
                "error": err
            })
            
            if not success:
                error_step = i
                error_type = self._categorize_error(err)
                failure_reason = f"Step {i} failed: {err}"
                break
        
        # =====================================================================
        # STAGE 4: Check Goal
        # =====================================================================
        goal_success = False
        failed_goals = []
        
        if failure_reason is None:
            if goal_spec:
                goal_success, failed_goals = simulator.check_goal(goal_spec)
            else:
                goal_success = True
        
        return {
            "success": goal_success,
            "goal_achieved": goal_success,
            "failed_goals": failed_goals,
            "steps_executed": len([e for e in executed if e["success"]]),
            "total_steps": len(steps),
            "api_calls": api_calls,
            "raw_plan": raw_plan,
            "nl_steps": nl_steps,
            "original_steps": original_steps,
            "final_steps": steps,
            "translation_failures": translation_failures,
            "failure_reason": failure_reason,
            "error_step": error_step,
            "error_type": error_type,
            "executed_trace": executed,
            "repair_history": repair_history,
            "validation_attempts": validation_attempts,
            "method": "repair_first"
        }
    
    def _build_contextual_prompt(self, instruction: str, simulator) -> str:
        """Build prompt with full state context."""
        location = simulator.get_agent_location()
        holding = simulator.get_holding()
        visible = simulator.visible_objects()
        
        # Get objects in all rooms
        all_objects_by_room = {}
        for obj_name, obj_props in simulator.state["objects"].items():
            room = obj_props["location"]
            if room != "agent":
                if room not in all_objects_by_room:
                    all_objects_by_room[room] = []
                all_objects_by_room[room].append(obj_name)
        
        room_info = []
        for room in ["kitchen", "bathroom", "bedroom", "living_room"]:
            objects = all_objects_by_room.get(room, [])
            room_info.append(f"  {room}: {', '.join(objects) if objects else 'empty'}")
        
        return f"""Current State:
- Location: {location}
- Holding: {holding if holding else 'nothing'}
- Visible here: {', '.join(visible) if visible else 'none'}

All Objects:
{chr(10).join(room_info)}

Task: {instruction}

IMPORTANT RULES:
1. You must "goto <room>" before you can interact with objects in that room
2. You must have empty hands to "pickup" (drop first if holding something)
3. Objects can only be used/toggled when you're in the same room

Actions: goto, pickup, drop, toggle, use
Format: One simple action per line (e.g., "goto bathroom")

Plan:"""
    
    def _build_repair_prompt(
        self, 
        failed_step: str, 
        error_msg: str, 
        simulator_state, 
        prefix_steps: List[str],
        original_instruction: str
    ) -> str:
        """Build prompt to repair a single failed step."""
        location = simulator_state.get_agent_location()
        holding = simulator_state.get_holding()
        visible = simulator_state.visible_objects()
        
        prefix_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(prefix_steps)) if prefix_steps else "  (none)"
        
        return f"""A step in your plan failed. Fix ONLY this step.

Original task: {original_instruction}

Steps executed successfully:
{prefix_str}

Current state after those steps:
- Location: {location}
- Holding: {holding if holding else 'nothing'}
- Visible objects: {', '.join(visible) if visible else 'none'}

FAILED STEP: {failed_step}
ERROR: {error_msg}

Rules reminder:
- "goto <room>" to move (kitchen, bathroom, bedroom, living_room)
- "pickup <object>" requires: empty hands AND object in current room
- "drop <object>" requires: holding that object
- "toggle/use <object>" requires: object in current room

Output ONLY the corrected action (e.g., "goto bathroom" or "drop cup"):"""
    
    def _parse_steps(self, raw_plan: str) -> List[str]:
        """Parse raw LLM output into clean step strings."""
        steps = []
        for line in raw_plan.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^[\d\.\-\*\•\)]+[\s\.)]*', '', line)
            line = line.strip()
            if line and len(line) > 2:
                steps.append(line)
        return steps
    
    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error for analysis."""
        error_msg = error_msg.lower()
        
        if "hand not empty" in error_msg:
            return "precondition_hands_full"
        elif "not holding" in error_msg:
            return "precondition_not_holding"
        elif "not in" in error_msg:
            return "precondition_wrong_location"
        elif "unknown" in error_msg or "invalid" in error_msg:
            return "invalid_target"
        else:
            return "execution_error"


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    from planner.llm_client import get_llm_client
    from simulator.symbolic_home import SymbolicHome
    
    print("=" * 60)
    print("REPAIR-FIRST PLANNER TEST")
    print("=" * 60)
    
    # Setup
    llm = get_llm_client()
    planner = RepairFirstPlanner(llm)
    sim = SymbolicHome()
    
    # Test task
    task = "wash hands"
    goal = ["hands_washed"]
    
    print(f"\nTask: {task}")
    print(f"Goal: {goal}")
    print(f"Initial state: {sim.get_state_description()}")
    print("\nExecuting with max 2 repairs...")
    print("-" * 40)
    
    result = planner.solve(task, sim, goal_spec=goal, max_repairs=2)
    
    print(f"\nRaw Plan:\n{result['raw_plan']}")
    print(f"\nOriginal Steps: {result['original_steps']}")
    print(f"Final Steps: {result['final_steps']}")
    
    if result['repair_history']:
        print(f"\nRepair History:")
        for repair in result['repair_history']:
            print(f"  Attempt {repair['attempt']}: Step {repair['step_index']}")
            print(f"    Original: {repair['original_step']}")
            print(f"    Error: {repair['error']}")
            print(f"    Repaired: {repair['repaired_action']}")
            print(f"    Success: {repair['success']}")
    
    print(f"\nExecution Trace:")
    for step in result['executed_trace']:
        status = "✓" if step['success'] else "✗"
        print(f"  {status} {step['action']} - {step.get('error', 'OK')}")
    
    print(f"\n" + "=" * 40)
    print(f"Steps executed: {result['steps_executed']}/{result['total_steps']}")
    print(f"Goal achieved: {result['goal_achieved']}")
    print(f"Success: {result['success']}")
    print(f"Failure reason: {result['failure_reason']}")
    print(f"API calls: {result['api_calls']} (target: ≤3)")
    print(f"Validation attempts: {result['validation_attempts']}")
    print("=" * 60)
