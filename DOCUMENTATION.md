# Language Planner: Technical Documentation

## Overview

This document provides comprehensive technical documentation for the Language Planner project, a research framework for evaluating LLM-based task planning strategies in symbolic environments.

---

## Architecture

The application follows a modular architecture with clear separation between the simulation layer, planning layer, and evaluation layer.

### Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Evaluation Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ tasks.json  │  │ evaluate.py │  │ failure_analysis│  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Planning Layer                        │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐  │
│  │ translator   │  │ huang_baseline │  │ repair_first│  │
│  │              │  │ contextual_... │  │             │  │
│  └──────────────┘  └────────────────┘  └─────────────┘  │
│                           │                              │
│                    ┌──────────────┐                      │
│                    │  llm_client  │                      │
│                    └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Simulation Layer                       │
│  ┌─────────────────┐  ┌────────────────────────────┐    │
│  │  action_space   │  │     symbolic_home          │    │
│  │  - VALID_ACTIONS│  │  - state management        │    │
│  │  - ROOMS        │  │  - is_valid / execute      │    │
│  │  - OBJECTS      │  │  - clone / reset           │    │
│  │  - parse_action │  │  - check_goal              │    │
│  └─────────────────┘  └────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## Simulation Layer

### Action Space (`simulator/action_space.py`)

The action space module defines the vocabulary of actions, rooms, and objects available in the simulation.

#### Constants

| Constant | Type | Description |
|----------|------|-------------|
| `VALID_ACTIONS` | set | {"goto", "pickup", "drop", "toggle", "use"} |
| `ROOMS` | list | ["kitchen", "bedroom", "bathroom", "living_room"] |
| `OBJECTS` | list | 15 household objects with defined locations |

#### Functions

**`parse_action(action_str: str) -> Tuple[Optional[str], Optional[str], Optional[str]]`**

Parses a raw action string and validates it against the defined vocabulary.

Parameters:
- `action_str`: A string like "goto kitchen" or "pickup cup"

Returns:
- `(action, argument, None)` on success
- `(None, None, error_message)` on failure

The function handles multi-word arguments (e.g., "living room" → "living_room") and validates that action-argument pairs are semantically valid (goto must target a room, pickup must target an object).

---

### Symbolic Home (`simulator/symbolic_home.py`)

The symbolic home module manages the complete state of the simulated environment.

#### State Structure

```python
state = {
    "agent": {
        "location": str,      # Current room
        "holding": str | None # Object in hand
    },
    "objects": {
        "object_name": {
            "location": str,  # Room name or "agent" if held
            "state": str      # Object-specific state
        }
    }
}
```

#### Class: SymbolicHome

**Constructor**

```python
def __init__(self, initial_state: Optional[Dict] = None)
```

Creates a new simulation environment. If no initial state is provided, the default home layout is used.

**Methods**

| Method | Description |
|--------|-------------|
| `reset()` | Restores the environment to its initial state |
| `clone()` | Creates a deep copy for speculative execution |
| `get_agent_location()` | Returns the current room name |
| `get_holding()` | Returns the held object name or None |
| `visible_objects()` | Returns list of objects in current room |
| `is_valid(action_str)` | Checks if action is valid in current state |
| `execute(action_str)` | Validates and executes an action |
| `check_goal(goal_spec)` | Evaluates goal predicates |

**Precondition Rules**

The `is_valid` method enforces the following preconditions:

| Action | Precondition |
|--------|--------------|
| goto | Always valid if room exists |
| pickup | Hand empty AND object in current room |
| drop | Currently holding the specified object |
| toggle | Object in current room |
| use | Object in current room OR currently holding it |

---

## Planning Layer

### Translator (`planner/translator.py`)

The translator module converts natural language action descriptions into canonical action strings.

#### Class: ActionTranslator

**Translation Process**

1. **Input Cleaning**: Remove filler words (the, a, an, to, please, etc.)
2. **Verb Matching**: Match verb against synonym dictionary
3. **Argument Extraction**: Match remaining words against object/room synonyms
4. **Fuzzy Fallback**: Apply difflib matching for variations

**Synonym Dictionaries**

```python
verb_synonyms = {
    "pickup": ["pick", "grab", "take", "get", "hold", "lift", "collect"],
    "goto": ["go", "move", "walk", "head", "travel", "enter", "visit"],
    "drop": ["put", "place", "set", "leave", "release"],
    "toggle": ["turn", "switch", "flip"],
    "use": ["use", "operate", "activate", "brush", "scrub"]
}

object_synonyms = {
    "cup": ["mug", "glass"],
    "faucet": ["tap", "sink", "water"],
    "toothbrush": ["teeth", "tooth"],
    # ... additional mappings
}
```

---

### LLM Client (`planner/llm_client.py`)

The LLM client provides a wrapper around the Groq API with rate limiting and usage tracking.

#### Class: GroqClient

**Constructor**

```python
def __init__(self, model: str = "llama-3.3-70b-versatile")
```

Initializes the client with the specified model. Requires `GROQ_API_KEY` environment variable.

**Methods**

| Method | Description |
|--------|-------------|
| `generate(prompt, max_tokens, temperature)` | Generate text with rate limiting |
| `get_stats()` | Return usage statistics |

**Rate Limiting**

The client enforces a minimum 3-second delay between API calls to respect Groq's rate limits (20 calls/minute on free tier).

---

### Huang Baseline (`planner/huang_baseline.py`)

The Huang Baseline implements zero-shot planning without environmental context.

#### Algorithm

1. Build prompt with task instruction only (no state)
2. Single LLM call to generate complete plan
3. Parse natural language steps
4. Translate to canonical actions
5. Execute open-loop (no validation)
6. Record success/failure

#### Prompt Template

```
Task: {instruction}

Generate a step-by-step plan to complete this task in a home environment.

Available actions:
- goto <room>: Move to a room
- pickup <object>: Pick up an object
- drop <object>: Put down an object
- toggle <object>: Turn something on/off
- use <object>: Use an object

Plan:
```

---

### Contextual Baseline (`planner/contextual_baseline.py`)

The Contextual Baseline extends Huang by including full state information in the prompt.

#### Key Difference

The prompt includes:
- Current location
- Currently held object
- Visible objects in current room
- Complete room-by-room object inventory

This context eliminates hallucination errors where the LLM assumes incorrect object locations.

#### Prompt Template

```
Current State:
- You are in: {location}
- Holding: {holding or 'nothing'}
- Objects visible here: {visible_objects}

Room Contents:
  kitchen: {kitchen_objects}
  bathroom: {bathroom_objects}
  bedroom: {bedroom_objects}
  living_room: {living_room_objects}

Task: {instruction}

Generate a step-by-step plan...
```

---

### RepairFirst Planner (`planner/repair_first.py`)

The RepairFirst Planner adds speculative validation and targeted repair to the contextual approach.

#### Algorithm

```
1. Generate initial plan with contextual prompt
2. Parse and translate to canonical actions
3. FOR each repair attempt (0 to max_repairs):
   a. Clone simulator
   b. FOR each step in plan:
      - Validate on clone
      - IF valid: execute on clone, continue
      - IF invalid: 
        * Generate repair prompt with error
        * Call LLM for single-step fix
        * Replace failed step
        * Break to restart validation
   c. IF no repairs needed: break
4. Execute validated plan on real simulator
5. Return results with repair history
```

#### Repair Prompt Template

```
A step in your plan failed. Fix ONLY this step.

Original task: {instruction}

Steps executed successfully:
  1. {step_1}
  2. {step_2}
  ...

Current state after those steps:
- Location: {location}
- Holding: {holding}
- Visible objects: {visible}

FAILED STEP: {failed_step}
ERROR: {error_msg}

Output ONLY the corrected action:
```

---

## Evaluation Layer

### Task Format (`eval/tasks.json`)

Tasks are defined in JSON format with the following schema:

```json
{
  "id": "M1",
  "instruction": "wash hands",
  "difficulty": "medium",
  "why_hard": "Requires navigation and multiple object interactions",
  "goal": ["hands_washed"],
  "expected_steps": 3
}
```

#### Difficulty Levels

| Level | Tasks | Characteristics |
|-------|-------|-----------------|
| Easy | 5 | 1-2 steps, single room |
| Medium | 10 | 3-4 steps, may require navigation |
| Hard | 5 | 5+ steps, multi-room, dependencies |

---

### Evaluation Runner (`eval/evaluate.py`)

The evaluation runner orchestrates benchmark execution and metric collection.

#### Metrics Collected

| Metric | Description |
|--------|-------------|
| success | Boolean goal achievement |
| steps_executed | Number of successfully executed steps |
| total_steps | Total steps in plan |
| api_calls | Number of LLM API calls |
| elapsed_seconds | Wall-clock time |
| failure_reason | Error message if failed |
| error_type | Categorized error (precondition, translation, etc.) |
| repair_history | List of repairs for RepairFirst |

#### Output Formats

Results are saved in three formats:
- `raw_results_{timestamp}.json`: Complete trace data
- `summary_{timestamp}.json`: Aggregated statistics
- `results_{timestamp}.csv`: Tabular format for analysis

---

## Goal Predicates

The simulator includes predefined goal predicates for common household tasks:

| Predicate | Condition |
|-----------|-----------|
| `hands_washed` | soap.state == "used" AND faucet.state == "on" |
| `teeth_brushed` | toothbrush.state == "used" |
| `coffee_made` | coffee_maker.state == "on" AND cup.state == "filled" |
| `cup_filled` | cup.state == "filled" |
| `lamp_on` | lamp.state == "on" |
| `lights_on` | light.state == "on" |
| `agent_in_kitchen` | agent.location == "kitchen" |
| `holding_cup` | agent.holding == "cup" |

---

## Error Taxonomy

The system categorizes execution errors for analysis:

| Error Type | Description | Example |
|------------|-------------|---------|
| `precondition_hands_full` | Cannot pickup while holding | "hand not empty" |
| `precondition_not_holding` | Cannot drop what you don't have | "not holding cup" |
| `precondition_wrong_location` | Object not in current room | "cup not in kitchen" |
| `invalid_target` | Unknown object or room | "unknown object: xyz" |
| `translation_failed` | Natural language not parseable | Could not translate step |
| `execution_error` | Other execution failure | Runtime error |

---

## API Reference

### Simulator

```python
from simulator import SymbolicHome, parse_action, ROOMS, OBJECTS

# Create environment
sim = SymbolicHome()

# Check action validity
valid, error = sim.is_valid("pickup cup")

# Execute action
success, error = sim.execute("goto bathroom")

# Clone for speculation
clone = sim.clone()

# Check goals
achieved, failed = sim.check_goal(["hands_washed"])
```

### Planner

```python
from planner import (
    ActionTranslator,
    get_llm_client,
    HuangBaseline,
    ContextualBaseline,
    RepairFirstPlanner
)

# Setup
llm = get_llm_client()
translator = ActionTranslator()

# Create planners
huang = HuangBaseline(llm, translator)
contextual = ContextualBaseline(llm, translator)
repair = RepairFirstPlanner(llm, translator)

# Solve task
result = repair.solve(
    instruction="wash hands",
    simulator=SymbolicHome(),
    goal_spec=["hands_washed"],
    max_repairs=2
)
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key for Groq LLM service |

### Rate Limiting

The default rate limit is 3 seconds between API calls. This can be adjusted by modifying `min_delay` in `GroqClient`:

```python
client = GroqClient()
client.min_delay = 1.0  # Faster if you have higher rate limits
```

---

## Extending the System

### Adding New Objects

1. Add object to `OBJECTS` list in `action_space.py`
2. Add default state to `DEFAULT_STATE` in `symbolic_home.py`
3. Add any special `use` effects in the `execute` method
4. Add synonyms to `object_synonyms` in `translator.py`

### Adding New Goal Predicates

Add a lambda function to `PREDICATES` in `symbolic_home.py`:

```python
PREDICATES["my_goal"] = lambda s: s["objects"]["item"]["state"] == "desired"
```

### Adding New Tasks

Add task definitions to `eval/tasks.json`:

```json
{
  "id": "N1",
  "instruction": "your task instruction",
  "difficulty": "medium",
  "goal": ["predicate_name"],
  "expected_steps": 4
}
```

---

## Limitations

The current implementation has several known limitations:

1. **Single-agent only**: The simulator supports only one robot agent.
2. **Discrete states**: Objects have simple discrete states rather than continuous properties.
3. **No partial observability**: The agent has full knowledge of all room contents.
4. **Fixed room topology**: The four-room layout is hardcoded.
5. **No object mobility**: Large objects (faucet, coffee_maker) cannot be moved.

These limitations are intentional to keep the simulation tractable while still demonstrating the core planning concepts.

This implementation prioritizes architectural clarity over full semantic fidelity. Known limitations include: (1) occasional goal/task misalignment in the benchmark, where formal predicates under-specify task intent; (2) inconsistent state semantics for the coffee maker (USE sets 'used', predicate checks 'on'); (3) deterministic difflib translation rather than the embedding-based approach of Huang et al. These issues affect absolute success rates but do not invalidate the comparative analysis, as all methods encounter the same simulator behavior. Future work would address these semantics and migrate to VirtualHome for direct comparison.

---

## References

- Huang, W., et al. (2022). Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents.
- SayCan (Ahn et al., 2022): Grounding language in robot affordances.
- Inner Monologue (Huang et al., 2022): Embodied reasoning with LLM feedback.
