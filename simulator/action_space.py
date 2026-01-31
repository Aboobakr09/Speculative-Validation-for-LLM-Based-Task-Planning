"""
Action Space Module for Home Robot Simulator.

Defines valid actions, rooms, and objects available in the simulation.
Provides parsing and validation for action strings.
"""

import re
from typing import Tuple, Optional


VALID_ACTIONS = {"goto", "pickup", "drop", "toggle", "use"}

ROOMS = ["kitchen", "bedroom", "bathroom", "living_room"]

# Objects that can be interacted with (expandable)
OBJECTS = [
    "cup", "plate", "soap", "towel", "faucet", "light", 
    "coffee_maker", "remote", "book", "phone", "keys",
    "toothbrush", "lamp", "blanket", "pillow"
]

# Actions that require specific argument types
ACTION_REQUIREMENTS = {
    "goto": {"requires": "room", "valid_targets": ROOMS},
    "pickup": {"requires": "object", "valid_targets": OBJECTS},
    "drop": {"requires": "object", "valid_targets": OBJECTS},
    "toggle": {"requires": "object", "valid_targets": OBJECTS},
    "use": {"requires": "object", "valid_targets": OBJECTS},
}

def parse_action(action_str: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse and validate an action string.
    
    Args:
        action_str: A string like "goto kitchen" or "pickup cup"
        
    Returns:
        Tuple of (action, argument, error_message)
        - On success: (action, argument, None)
        - On failure: (None, None, error_message)
        
    Examples:
        >>> parse_action("goto kitchen")
        ('goto', 'kitchen', None)
        
        >>> parse_action("pickup cup")
        ('pickup', 'cup', None)
        
        >>> parse_action("fly away")
        (None, None, "Unknown action 'fly'. Valid actions: goto, pickup, drop, toggle, use")
        
        >>> parse_action("goto mars")
        (None, None, "Invalid target 'mars' for 'goto'. Valid rooms: kitchen, bedroom, bathroom, living_room")
    """
    # Handle empty or whitespace-only input
    if not action_str or not action_str.strip():
        return None, None, "Empty action string"
    
    # Normalize: lowercase and strip whitespace
    action_str = action_str.strip().lower()
    
    # Split into tokens (handle multiple spaces)
    tokens = action_str.split()
    
    if len(tokens) < 2:
        return None, None, f"Malformed action '{action_str}'. Expected format: 'action argument' (e.g., 'goto kitchen')"
    
    if len(tokens) > 2:
        # Handle compound arguments like "living_room" written as "living room"
        action = tokens[0]
        # Try to join remaining tokens with underscore
        argument = "_".join(tokens[1:])
    else:
        action, argument = tokens[0], tokens[1]
    
    # Validate action
    if action not in VALID_ACTIONS:
        valid_list = ", ".join(sorted(VALID_ACTIONS))
        return None, None, f"Unknown action '{action}'. Valid actions: {valid_list}"
    
    # Validate argument based on action type
    requirements = ACTION_REQUIREMENTS[action]
    valid_targets = requirements["valid_targets"]
    
    if argument not in valid_targets:
        target_type = requirements["requires"]
        valid_list = ", ".join(valid_targets)
        return None, None, f"Invalid target '{argument}' for '{action}'. Valid {target_type}s: {valid_list}"
    
    return action, argument, None


def is_valid_action(action_str: str) -> bool:
    """
    Quick validity check for an action string.
    
    Args:
        action_str: The action string to validate
        
    Returns:
        True if valid, False otherwise
    """
    _, _, error = parse_action(action_str)
    return error is None


def get_action_help() -> str:
    """
    Return a help string describing all valid actions and their targets.
    """
    lines = ["Available Actions:", "=" * 40]
    
    for action in sorted(VALID_ACTIONS):
        req = ACTION_REQUIREMENTS[action]
        targets = ", ".join(req["valid_targets"][:5])
        if len(req["valid_targets"]) > 5:
            targets += ", ..."
        lines.append(f"  {action} <{req['requires']}>")
        lines.append(f"    Valid targets: {targets}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    # Run basic tests
    print("Testing action_space module...")
    print("=" * 50)
    
    # Test valid actions
    test_cases = [
        ("goto kitchen", ("goto", "kitchen", None)),
        ("pickup cup", ("pickup", "cup", None)),
        ("drop plate", ("drop", "plate", None)),
        ("toggle faucet", ("toggle", "faucet", None)),
        ("use soap", ("use", "soap", None)),
        ("goto living room", ("goto", "living_room", None)),  # Space handling
    ]
    
    print("\n✓ Valid action tests:")
    for action_str, expected in test_cases:
        result = parse_action(action_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} parse_action('{action_str}') = {result}")
    
    # Test invalid actions
    invalid_cases = [
        "fly away",       # Unknown action
        "goto mars",      # Invalid room
        "pickup dragon",  # Invalid object
        "",               # Empty string
        "goto",           # Missing argument
    ]
    
    print("\n✗ Invalid action tests (should return errors):")
    for action_str in invalid_cases:
        action, arg, error = parse_action(action_str)
        status = "✓" if error is not None else "✗"
        print(f"  {status} parse_action('{action_str}') -> Error: {error}")
    
    print("\n" + get_action_help())
    print("\n" + "=" * 50)
    print("All tests completed!")
