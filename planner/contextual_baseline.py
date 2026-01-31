"""
Contextual Open-Loop Baseline (Control Method).

Same as Huang baseline BUT with state context in prompt.
This is the CONTROL to prove that better prompting alone isn't enough.

Key characteristics:
1. Single LLM call (zero-shot)
2. YES environment state in prompt (reduces hallucinations)
3. Open-loop execution (no mid-execution validation)
4. No repair mechanism
"""

import re
from typing import Dict, List, Optional, Any
from planner.translator import ActionTranslator


class ContextualBaseline:
    """
    Contextual Open-Loop Baseline - the control method.
    
    Improvement over Huang: Includes current state in prompt.
    Still limited: No precondition checking during execution.
    
    This isolates the effect of "better prompting" from validation/repair.
    
    Usage:
        llm = get_llm_client()
        baseline = ContextualBaseline(llm)
        result = baseline.solve("wash hands", SymbolicHome())
    """
    
    def __init__(self, llm_client, translator: Optional[ActionTranslator] = None):
        """
        Initialize Contextual baseline planner.
        
        Args:
            llm_client: LLM client with generate() method
            translator: ActionTranslator instance (creates one if not provided)
        """
        self.llm = llm_client
        self.translator = translator or ActionTranslator()
    
    def solve(self, instruction: str, simulator, goal_spec: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute Contextual Open-Loop planning method.
        
        Args:
            instruction: Natural language task description
            simulator: SymbolicHome instance
            goal_spec: Optional list of goal predicates to check
        
        Returns:
            Dict with same keys as HuangBaseline for comparison
        """
        # Reset simulator to initial state
        simulator.reset()
        
        # =====================================================================
        # STAGE 1: Single LLM call WITH STATE CONTEXT (key difference)
        # =====================================================================
        prompt = self._build_prompt(instruction, simulator)
        raw_plan = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        
        # =====================================================================
        # STAGE 2: Parse natural language steps
        # =====================================================================
        nl_steps = self._parse_steps(raw_plan)
        
        # =====================================================================
        # STAGE 3: Translate to API actions
        # =====================================================================
        translated_steps = []
        translation_failures = []
        
        for i, nl_step in enumerate(nl_steps):
            action, conf, method = self.translator.translate(nl_step)
            if action:
                translated_steps.append(action)
            else:
                translation_failures.append({
                    "step_index": i,
                    "nl_step": nl_step,
                    "reason": "translation_failed"
                })
        
        # =====================================================================
        # STAGE 4: Open-loop execution (still no validation!)
        # =====================================================================
        executed = []
        failure_reason = None
        error_step = None
        error_type = None
        
        for i, step in enumerate(translated_steps):
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
                break  # Still open-loop: stop on first failure
        
        # =====================================================================
        # STAGE 5: Check goal
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
            "total_steps": len(translated_steps),
            "api_calls": 1,  # Still 1 call (same as Huang)
            "raw_plan": raw_plan,
            "nl_steps": nl_steps,
            "translated_steps": translated_steps,
            "translation_failures": translation_failures,
            "failure_reason": failure_reason,
            "error_step": error_step,
            "error_type": error_type,
            "executed_trace": executed,
            "method": "contextual_open_loop"
        }
    
    def _build_prompt(self, instruction: str, simulator) -> str:
        """
        Context-aware prompt: Includes current state to reduce hallucinations.
        Key improvement over Huang - LLM knows what objects are visible.
        """
        location = simulator.get_agent_location()
        holding = simulator.get_holding()
        visible = simulator.visible_objects()
        
        # Get objects in all rooms for context
        all_objects_by_room = {}
        for obj_name, obj_props in simulator.state["objects"].items():
            room = obj_props["location"]
            if room != "agent":  # Skip held objects
                if room not in all_objects_by_room:
                    all_objects_by_room[room] = []
                all_objects_by_room[room].append(obj_name)
        
        room_info = []
        for room in ["kitchen", "bathroom", "bedroom", "living_room"]:
            objects = all_objects_by_room.get(room, [])
            room_info.append(f"  {room}: {', '.join(objects) if objects else 'empty'}")
        
        return f"""Current State:
- You are in: {location}
- Holding: {holding if holding else 'nothing'}
- Objects visible here: {', '.join(visible) if visible else 'none'}

Room Contents:
{chr(10).join(room_info)}

Task: {instruction}

Generate a step-by-step plan. You must navigate to objects before using them.

Available actions:
- goto <room>: Move to kitchen, bathroom, bedroom, or living_room
- pickup <object>: Pick up (must be in same room, hands empty)
- drop <object>: Put down (must be holding it)
- toggle <object>: Turn on/off (must be in same room)
- use <object>: Use object (must be in same room)

Requirements:
- One action per line
- Be precise: "goto bathroom" not "go to the bathroom"
- No explanations or numbering

Plan:"""
    
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
    print("CONTEXTUAL BASELINE TEST")
    print("=" * 60)
    
    # Setup
    llm = get_llm_client()
    baseline = ContextualBaseline(llm)
    sim = SymbolicHome()
    
    # Test task
    task = "wash hands"
    goal = ["hands_washed"]
    
    print(f"\nTask: {task}")
    print(f"Goal: {goal}")
    print(f"Initial state: {sim.get_state_description()}")
    print("\nExecuting...")
    print("-" * 40)
    
    result = baseline.solve(task, sim, goal_spec=goal)
    
    print(f"\nRaw Plan:\n{result['raw_plan']}")
    print(f"\nTranslated Steps: {result['translated_steps']}")
    print(f"\nExecution Trace:")
    for step in result['executed_trace']:
        status = "✓" if step['success'] else "✗"
        print(f"  {status} {step['action']} - {step.get('error', 'OK')}")
    
    print(f"\n" + "=" * 40)
    print(f"Steps executed: {result['steps_executed']}/{result['total_steps']}")
    print(f"Goal achieved: {result['goal_achieved']}")
    print(f"Success: {result['success']}")
    print(f"Failure reason: {result['failure_reason']}")
    print(f"Error type: {result['error_type']}")
    print(f"API calls: {result['api_calls']}")
    print("=" * 60)
