"""
Huang et al. (2022) Faithful Reimplementation.
Zero-shot planning with semantic translation, no feedback loops.

Key characteristics:
1. Single LLM call (zero-shot)
2. No environment state in prompt
3. Open-loop execution (no mid-execution validation)
4. Semantic translation layer (our difflib translator)
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from planner.translator import ActionTranslator


class HuangBaseline:
    """
    Faithful reimplementation of Huang et al. 2022 Language Planner.
    
    This is the BASELINE we are trying to beat. It has known weaknesses:
    - No state awareness (hallucinates object locations)
    - No precondition checking (crashes on invalid actions)
    - No recovery mechanism (stops on first failure)
    
    Usage:
        llm = get_llm_client()
        baseline = HuangBaseline(llm)
        result = baseline.solve("wash hands", SymbolicHome())
    """
    
    def __init__(self, llm_client, translator: Optional[ActionTranslator] = None):
        """
        Initialize Huang baseline planner.
        
        Args:
            llm_client: LLM client with generate() method
            translator: ActionTranslator instance (creates one if not provided)
        """
        self.llm = llm_client
        self.translator = translator or ActionTranslator()
    
    def solve(self, instruction: str, simulator, goal_spec: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute Huang et al. planning method.
        
        Args:
            instruction: Natural language task description
            simulator: SymbolicHome instance
            goal_spec: Optional list of goal predicates to check
        
        Returns:
            Dict with keys:
                - success: bool (goal achieved without errors)
                - steps_executed: int (successful steps)
                - total_steps: int (planned steps)
                - api_calls: int (always 1 for Huang)
                - raw_plan: str (LLM output)
                - nl_steps: list (parsed steps)
                - translated_steps: list (action strings)
                - translation_failures: list (failed translations)
                - failure_reason: str or None
                - error_step: int or None
                - error_type: str or None
                - executed_trace: list (execution log)
        """
        # Reset simulator to initial state
        simulator.reset()
        
        # =====================================================================
        # STAGE 1: Single LLM call (the core of Huang method)
        # =====================================================================
        prompt = self._build_prompt(instruction)
        raw_plan = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        
        # =====================================================================
        # STAGE 2: Parse natural language steps
        # =====================================================================
        nl_steps = self._parse_steps(raw_plan)
        
        # =====================================================================
        # STAGE 3: Translate to API actions (Huang's semantic matching layer)
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
        # STAGE 4: Open-loop execution (no validation, execute until crash)
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
                # Categorize failure for analysis
                error_step = i
                error_type = self._categorize_error(err)
                failure_reason = f"Step {i} failed: {err}"
                break  # Open-loop: stop on first failure
        
        # =====================================================================
        # STAGE 5: Check goal (if goal_spec provided)
        # =====================================================================
        goal_success = False
        failed_goals = []
        
        if failure_reason is None:
            if goal_spec:
                goal_success, failed_goals = simulator.check_goal(goal_spec)
            else:
                # No goal spec = success if no execution errors
                goal_success = True
        
        return {
            "success": goal_success,
            "goal_achieved": goal_success,
            "failed_goals": failed_goals,
            "steps_executed": len([e for e in executed if e["success"]]),
            "total_steps": len(translated_steps),
            "api_calls": 1,  # Always 1 for Huang baseline
            "raw_plan": raw_plan,
            "nl_steps": nl_steps,
            "translated_steps": translated_steps,
            "translation_failures": translation_failures,
            "failure_reason": failure_reason,
            "error_step": error_step,
            "error_type": error_type,
            "executed_trace": executed
        }
    
    def _build_prompt(self, instruction: str) -> str:
        """
        Huang et al. style prompt: Task only, NO state context.
        This is intentionally limited - it's what we're trying to improve on.
        """
        return f"""Task: {instruction}

Generate a step-by-step plan to complete this task in a home environment.

Available actions:
- goto <room>: Move to a room (kitchen, bedroom, bathroom, living_room)
- pickup <object>: Pick up an object
- drop <object>: Put down an object you're holding
- toggle <object>: Turn something on/off
- use <object>: Use an object

Requirements:
- One action per line
- Use simple language (e.g., "goto kitchen", "pickup cup")
- Do not include explanations or numbering
- Be concise

Plan:"""
    
    def _parse_steps(self, raw_plan: str) -> List[str]:
        """Parse raw LLM output into clean step strings."""
        steps = []
        for line in raw_plan.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # Remove leading numbers/bullets (1., 1), -, *, •, etc.)
            line = re.sub(r'^[\d\.\-\*\•\)]+[\s\.)]*', '', line)
            line = line.strip()
            # Skip empty or very short lines
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
    print("HUANG BASELINE TEST")
    print("=" * 60)
    
    # Setup
    llm = get_llm_client()
    baseline = HuangBaseline(llm)
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
