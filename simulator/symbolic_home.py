"""
Symbolic Home Environment for Language Planner.

Manages robot state in a simulated home environment with rooms and objects.
Supports cloning for speculative validation and goal checking.
"""

import copy
from typing import Dict, List, Tuple, Optional, Any, Callable

from .action_space import parse_action, ROOMS, OBJECTS


# =============================================================================
# Default Home Layout
# =============================================================================

DEFAULT_STATE = {
    "agent": {
        "location": "kitchen",
        "holding": None
    },
    "objects": {
        # Kitchen objects
        "cup": {"location": "kitchen", "state": "empty"},
        "plate": {"location": "kitchen", "state": "clean"},
        "coffee_maker": {"location": "kitchen", "state": "off"},
        
        # Bathroom objects
        "soap": {"location": "bathroom", "state": "unused"},
        "towel": {"location": "bathroom", "state": "dry"},
        "faucet": {"location": "bathroom", "state": "off"},
        "toothbrush": {"location": "bathroom", "state": "unused"},
        
        # Bedroom objects
        "lamp": {"location": "bedroom", "state": "off"},
        "blanket": {"location": "bedroom", "state": "folded"},
        "pillow": {"location": "bedroom", "state": "on_bed"},
        
        # Living room objects
        "remote": {"location": "living_room", "state": "off"},
        "book": {"location": "living_room", "state": "closed"},
        "light": {"location": "living_room", "state": "off"},
        
        # Portable objects (can start anywhere)
        "phone": {"location": "living_room", "state": "off"},
        "keys": {"location": "kitchen", "state": "unused"},
    }
}


# =============================================================================
# Goal Predicates
# =============================================================================

PREDICATES: Dict[str, Callable[[Dict], bool]] = {
    # Hygiene goals
    "hands_washed": lambda s: (
        s["objects"]["soap"]["state"] == "used" and 
        s["objects"]["faucet"]["state"] == "on"
    ),
    "teeth_brushed": lambda s: s["objects"]["toothbrush"]["state"] == "used",
    
    # Kitchen goals
    "coffee_made": lambda s: (
        s["objects"]["coffee_maker"]["state"] == "used" and  # Fixed: was checking 'on'
        s["objects"]["cup"]["state"] == "filled"
    ),
    "cup_filled": lambda s: s["objects"]["cup"]["state"] == "filled",
    
    # Room state goals
    "lights_on": lambda s: s["objects"]["light"]["state"] == "on",
    "lights_off": lambda s: s["objects"]["light"]["state"] == "off",
    "lamp_on": lambda s: s["objects"]["lamp"]["state"] == "on",
    
    # Object location goals
    "cup_in_kitchen": lambda s: s["objects"]["cup"]["location"] == "kitchen",
    "cup_in_living_room": lambda s: s["objects"]["cup"]["location"] == "living_room",
    "phone_in_bedroom": lambda s: s["objects"]["phone"]["location"] == "bedroom",
    
    # Agent state goals
    "agent_in_kitchen": lambda s: s["agent"]["location"] == "kitchen",
    "agent_in_bathroom": lambda s: s["agent"]["location"] == "bathroom",
    "agent_in_bedroom": lambda s: s["agent"]["location"] == "bedroom",
    "agent_in_living_room": lambda s: s["agent"]["location"] == "living_room",
    "hands_empty": lambda s: s["agent"]["holding"] is None,
    "holding_cup": lambda s: s["agent"]["holding"] == "cup",
    "holding_phone": lambda s: s["agent"]["holding"] == "phone",
}


# =============================================================================
# SymbolicHome Class
# =============================================================================

class SymbolicHome:
    """
    Symbolic simulation of a home environment for robot planning.
    
    The robot can navigate between rooms, pick up and drop objects,
    toggle object states, and use objects.
    
    Attributes:
        state: Current state dictionary with "agent" and "objects" keys
    """
    
    def __init__(self, initial_state: Optional[Dict] = None):
        """
        Initialize the home environment.
        
        Args:
            initial_state: Optional custom initial state. Uses DEFAULT_STATE if None.
        """
        self._initial_state = copy.deepcopy(initial_state or DEFAULT_STATE)
        self.state = copy.deepcopy(self._initial_state)
    
    def reset(self) -> Dict:
        """
        Reset the environment to its initial state.
        
        Returns:
            The reset state dictionary
        """
        self.state = copy.deepcopy(self._initial_state)
        return self.state
    
    def clone(self) -> "SymbolicHome":
        """
        Create a deep copy of this environment for speculative execution.
        
        Returns:
            A new SymbolicHome instance with identical state
        """
        cloned = SymbolicHome.__new__(SymbolicHome)
        cloned._initial_state = copy.deepcopy(self._initial_state)
        cloned.state = copy.deepcopy(self.state)
        return cloned
    
    def get_agent_location(self) -> str:
        """Get the agent's current room."""
        return self.state["agent"]["location"]
    
    def get_holding(self) -> Optional[str]:
        """Get the object the agent is currently holding, or None."""
        return self.state["agent"]["holding"]
    
    def visible_objects(self) -> List[str]:
        """
        Get list of objects visible in the agent's current room.
        
        Returns:
            List of object names in the current room (excluding held objects)
        """
        current_room = self.get_agent_location()
        visible = []
        
        for obj_name, obj_props in self.state["objects"].items():
            if obj_props["location"] == current_room:
                visible.append(obj_name)
        
        return sorted(visible)
    
    def get_state_description(self) -> str:
        """
        Get a human-readable description of the current state.
        Useful for LLM context injection.
        """
        location = self.get_agent_location()
        holding = self.get_holding()
        visible = self.visible_objects()
        
        lines = [
            f"Current location: {location}",
            f"Visible objects: [{', '.join(visible) if visible else 'none'}]",
            f"Holding: {holding if holding else 'nothing'}",
        ]
        
        return "\n".join(lines)
    
    # =========================================================================
    # Action Validation
    # =========================================================================
    
    def is_valid(self, action_str: str) -> Tuple[bool, Optional[str]]:
        """
        Check if an action is valid in the current state.
        
        Args:
            action_str: Action string like "goto kitchen" or "pickup cup"
            
        Returns:
            Tuple of (is_valid, error_message)
            - On valid: (True, None)
            - On invalid: (False, error_message)
        """
        action, arg, err = parse_action(action_str)
        if err:
            return False, err
        
        agent_loc = self.state["agent"]["location"]
        holding = self.state["agent"]["holding"]
        
        if action == "pickup":
            if holding is not None:
                return False, "hand not empty"
            if self.state["objects"][arg]["location"] != agent_loc:
                return False, f"{arg} not in {agent_loc}"
            return True, None
        
        elif action == "drop":
            if holding != arg:
                return False, f"not holding {arg}"
            return True, None
        
        elif action == "goto":
            return True, None
        
        elif action == "toggle":
            obj_loc = self.state["objects"][arg]["location"]
            if obj_loc != agent_loc:
                return False, f"{arg} not in {agent_loc} (it's in {obj_loc})"
            return True, None
        
        elif action == "use":
            obj_loc = self.state["objects"][arg]["location"]
            # Can use object if it's in current room OR if holding it
            if obj_loc != agent_loc and holding != arg:
                return False, f"{arg} not in {agent_loc} (it's in {obj_loc})"
            return True, None
        
        return False, f"unhandled action {action}"
    
    # =========================================================================
    # Action Execution
    # =========================================================================
    
    def execute(self, action_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate and execute an action, updating the state.
        
        Args:
            action_str: Action string like "goto kitchen" or "pickup cup"
            
        Returns:
            Tuple of (success, error_message)
            - On success: (True, None)
            - On failure: (False, error_message)
        """
        # Validate first
        is_valid, error = self.is_valid(action_str)
        if not is_valid:
            return False, error
        
        # Parse action (we know it's valid now)
        action, argument, _ = parse_action(action_str)
        
        # Execute based on action type
        if action == "goto":
            self.state["agent"]["location"] = argument
        
        elif action == "pickup":
            # Move object to "agent" (special location meaning held)
            self.state["objects"][argument]["location"] = "agent"
            self.state["agent"]["holding"] = argument
        
        elif action == "drop":
            # Move object to current room
            current_room = self.get_agent_location()
            self.state["objects"][argument]["location"] = current_room
            self.state["agent"]["holding"] = None
        
        elif action == "toggle":
            # Flip between on/off states
            current_state = self.state["objects"][argument]["state"]
            if current_state == "on":
                self.state["objects"][argument]["state"] = "off"
            elif current_state == "off":
                self.state["objects"][argument]["state"] = "on"
            else:
                # For non-on/off states, just toggle to "on"
                self.state["objects"][argument]["state"] = "on"
        
        elif action == "use":
            # Mark object as used and apply special effects
            obj_name = argument
            self.state["objects"][obj_name]["state"] = "used"
            
            # Special effects for certain objects
            if obj_name == "coffee_maker":
                # Using coffee maker fills the cup if cup is in kitchen
                if self.state["objects"]["cup"]["location"] == "kitchen":
                    self.state["objects"]["cup"]["state"] = "filled"
            
            elif obj_name == "faucet":
                # Using faucet turns it on AND fills cup if holding it
                self.state["objects"]["faucet"]["state"] = "on"
                if self.state["agent"]["holding"] == "cup":
                    self.state["objects"]["cup"]["state"] = "filled"
            
            elif obj_name == "cup":
                # Using cup at faucet fills it (if faucet is on)
                if (self.state["agent"]["location"] == "bathroom" and 
                    self.state["objects"]["faucet"]["state"] == "on"):
                    self.state["objects"]["cup"]["state"] = "filled"
        
        return True, None
    
    # =========================================================================
    # Goal Checking
    # =========================================================================
    
    def check_goal(self, goal_spec: str | List[str]) -> Tuple[bool, List[str]]:
        """
        Check if goal predicate(s) are satisfied.
        
        Args:
            goal_spec: Single goal string or list of goal strings
            
        Returns:
            Tuple of (all_satisfied, list_of_failed_goals)
        """
        if isinstance(goal_spec, str):
            goal_spec = [goal_spec]
        
        failed_goals = []
        
        for goal in goal_spec:
            if goal not in PREDICATES:
                failed_goals.append(f"{goal} (unknown predicate)")
                continue
            
            predicate = PREDICATES[goal]
            if not predicate(self.state):
                failed_goals.append(goal)
        
        return len(failed_goals) == 0, failed_goals
    
    def get_available_goals(self) -> List[str]:
        """Return list of all available goal predicates."""
        return list(PREDICATES.keys())


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    print("Testing SymbolicHome...")
    print("=" * 60)
    
    # Test initialization
    home = SymbolicHome()
    print(f"\n✓ Initial state:")
    print(f"  {home.get_state_description()}")
    
    # Test visible_objects
    print(f"\n✓ Visible objects in kitchen: {home.visible_objects()}")
    
    # Test is_valid
    print("\n✓ Validation tests:")
    test_actions = [
        ("goto bathroom", True),
        ("pickup cup", True),
        ("pickup soap", False),  # soap is in bathroom
        ("drop cup", False),     # not holding cup
    ]
    for action, expected_valid in test_actions:
        valid, err = home.is_valid(action)
        status = "✓" if valid == expected_valid else "✗"
        print(f"  {status} is_valid('{action}'): {valid} - {err if err else 'OK'}")
    
    # Test execute sequence
    print("\n✓ Execution sequence:")
    actions = [
        "pickup cup",
        "goto bathroom",
        "drop cup",
        "toggle faucet",
        "use soap",
    ]
    for action in actions:
        success, err = home.execute(action)
        status = "✓" if success else "✗"
        print(f"  {status} execute('{action}'): {'OK' if success else err}")
    
    print(f"\n  State after sequence:")
    print(f"  {home.get_state_description()}")
    
    # Test clone
    print("\n✓ Clone test:")
    cloned = home.clone()
    cloned.execute("goto kitchen")
    print(f"  Original location: {home.get_agent_location()}")
    print(f"  Cloned location: {cloned.get_agent_location()}")
    assert home.get_agent_location() != cloned.get_agent_location(), "Clone should be independent!"
    print("  ✓ Clone is independent of original")
    
    # Test reset
    print("\n✓ Reset test:")
    home.reset()
    print(f"  After reset: {home.get_state_description()}")
    
    # Test goal checking
    print("\n✓ Goal checking:")
    home.execute("goto bathroom")
    home.execute("toggle faucet")
    home.execute("use soap")
    satisfied, failed = home.check_goal("hands_washed")
    print(f"  hands_washed: {satisfied} (failed: {failed})")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
