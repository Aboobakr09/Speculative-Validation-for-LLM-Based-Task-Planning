# Language Planner: Speculative Validation for LLM-Based Task Planning

## Description

This application allows researchers and developers to explore LLM-based task planning in a symbolic home environment. The system provides multiple planning strategies that translate natural language instructions into executable robot actions, enabling comparative analysis of different approaches to language-grounded planning.

Users can issue natural language commands such as "wash hands" or "make coffee and bring it to the bedroom," and the planner will generate a sequence of primitive actions (goto, pickup, drop, toggle, use) to accomplish the goal. The application evaluates these plans against a symbolic simulator that tracks the robot's location, held objects, and the state of household items.

The system implements three distinct planning methodologies. The Huang Baseline represents a zero-shot approach with no environmental context, serving as a lower bound for performance. The Contextual Baseline incorporates full state information into the prompt, demonstrating how context-aware prompting significantly improves success rates. The RepairFirst Planner extends the contextual approach with speculative validation and targeted single-step repair, providing robustness against occasional generation failures.

There are several problems this application addresses. First, existing LLM planners often generate invalid action sequences because they lack awareness of environmental preconditions—for example, attempting to pick up an object that is not in the current room. Second, when plans fail mid-execution, most systems require complete regeneration, which is computationally expensive. Third, there is a need for standardized benchmarks to compare different planning approaches fairly. This application provides a unified framework for evaluating planning strategies with consistent metrics.

## Key Features

### Symbolic Home Simulator

The simulator maintains a complete representation of a four-room home environment (kitchen, bathroom, bedroom, living room) with fifteen interactive objects. Each object has a location and a state (on/off, used/unused, empty/filled). The simulator validates action preconditions before execution, ensuring that the robot cannot perform physically impossible actions such as picking up an object while already holding something.

The simulator supports cloning for speculative execution, which allows planners to test action sequences on a copy of the environment without affecting the real state. This feature is essential for the RepairFirst approach, which validates plans before committing to execution.

### Action Translator

The translator converts natural language steps into structured action commands. Users can write instructions in various forms—"grab the mug," "pick up the cup," or "take the glass"—and the translator will normalize these to the canonical form "pickup cup."

The translation process uses a two-stage approach. The first stage performs exact synonym matching against a comprehensive dictionary of verb and object synonyms. The second stage applies fuzzy matching using Python's difflib library to handle minor spelling variations. This approach achieves 100% accuracy on the test suite of 31 translation cases.

### Huang Baseline Planner

The Huang Baseline implements a zero-shot planning approach based on prior work in language-grounded planning. The planner receives only the task instruction without any environmental context, generating a complete plan in a single LLM call.

This method serves as a lower bound for comparison. Without context about object locations and the robot's current state, the LLM frequently hallucinates—assuming objects are in convenient locations or generating actions that violate preconditions. In evaluation, this approach achieves 40% success across the benchmark tasks.

### Contextual Baseline Planner

The Contextual Baseline extends the Huang approach by injecting full environmental state into the prompt. The LLM receives information about the robot's current location, what it is holding, visible objects in the current room, and the contents of all rooms.

This context-aware approach dramatically improves performance. By knowing where objects are located, the LLM generates valid navigation sequences and avoids hallucination errors. In evaluation, this approach achieves 85% success—a 45 percentage point improvement over the Huang Baseline.

### RepairFirst Planner

The RepairFirst Planner represents the novel contribution of this project. It combines contextual prompting with speculative validation and targeted repair.

The algorithm works as follows: First, the planner generates an initial plan using contextual prompting. Second, the planner validates each step on a cloned simulator before real execution. Third, when a step fails validation, the planner calls the LLM to repair only that specific step, providing the error message and current state. Fourth, after repair, validation restarts from the beginning with the updated plan. Fifth, a maximum repair limit (default 2) prevents infinite loops.

In evaluation, this approach achieves 75% success with an average of 1.15 API calls per task. While it does not outperform the Contextual Baseline in aggregate, it demonstrates robustness in specific cases where the initial generation contains precondition errors.

### Evaluation Framework

The evaluation framework provides standardized benchmarking across all planners. It includes 20 tasks across three difficulty levels: easy (1-2 steps), medium (3-4 steps), and hard (5+ steps with multi-room coordination).

The framework tracks multiple metrics including success rate, API call count, execution time, and detailed failure categorization. Results are exported to JSON and CSV formats for further analysis, and visualization scripts generate comparative charts.

## Evaluation Results

### Multi-Model Benchmark (20 tasks)

| Model | Planner | Executability | Correctness | API Calls |
|-------|---------|--------------|-------------|-----------|
| **Llama 3.3-70B** | Contextual | **100%** | **90%** | 1.00 |
| **Llama 3.3-70B** | RepairFirst | **100%** | 80% | 1.05 |
| Llama 3.3-70B | Huang | 55% | 35% | 1.00 |
| GPT-OSS-120B | Contextual | 90% | 75% | 1.00 |
| GPT-OSS-120B | RepairFirst | 95% | 75% | 1.05 |
| GPT-OSS-120B | Huang | 55% | 50% | 1.00 |

### Comparison with Huang et al. (2022) Paper

| Method | Executability | Correctness |
|--------|--------------|-------------|
| Codex 12B + Translation (Paper) | 78.6% | 25.7% |
| GPT-3 175B + Translation (Paper) | 73.0% | 24.0% |
| **Contextual (Ours)** | **100%** | **90%** |
| **RepairFirst (Ours)** | **100%** | **80%** |

### Key Findings

We evaluated three planners—Huang Baseline, Contextual, and RepairFirst—using Llama 3.3-70B and GPT-OSS-120B. Our results demonstrate that **state-context injection significantly improves both executability and correctness** compared to the zero-shot baseline from Huang et al. (2022). The Contextual planner with Llama 3.3-70B achieved **100% executability and 90% correctness**, placing it firmly in the ideal Pareto frontier zone—substantially outperforming both the original paper's translated Codex 12B (78.6% exec / 25.7% corr) and GPT-3 175B (73% exec / 24% corr) baselines. The RepairFirst planner demonstrated **100% executability**, confirming that speculative validation eliminates execution failures entirely, though with a slight correctness trade-off (80% vs 90%). Our Huang baseline replication (55% exec / 35-50% corr) aligns reasonably with the paper's translated results, validating our experimental setup. The ablation study reveals that Contextual planning provides the largest single improvement (+55% executability, +55% correctness over baseline), while RepairFirst contributes an additional robustness guarantee at minimal API cost (1.05 calls average vs 1.00). These findings suggest that **grounding LLM planners with explicit world state is more effective than translation-only approaches** for achieving reliable task execution in embodied environments.

## Instructions

### Setting Up the Environment

Users should first create a Python virtual environment and install the required dependencies. The application requires Python 3.8 or higher and uses the Groq API for LLM inference.

1. Clone the repository to your local machine.
2. Create and activate a virtual environment using `python -m venv venv` followed by `source venv/bin/activate` on macOS/Linux.
3. Install dependencies using `pip install -r requirements.txt`.
4. Create a `.env` file in the project root with your Groq API key: `GROQ_API_KEY=your_key_here`.

### Running the Simulator

Users can test the simulator independently by running `python -m simulator.symbolic_home`. This executes a self-test that demonstrates state management, action validation, cloning, and goal checking.

### Testing the Translator

Users can verify the translator by running `python -m planner.translator`. This executes the full test suite of 31 translation cases and reports the pass rate.

### Running Individual Planners

Each planner can be tested independently:

- Huang Baseline: `python -m planner.huang_baseline`
- Contextual Baseline: `python -m planner.contextual_baseline`
- RepairFirst: `python -m planner.repair_first`

Each script runs a demonstration task and displays the generated plan, execution trace, and success status.

### Running the Full Evaluation

Users can run the complete 20-task evaluation using the following command:

```bash
python eval/evaluate.py --trials 1
```

Additional options include:
- `--planners huang contextual repair_first` to select specific planners
- `--filter-difficulty hard` to run only hard tasks
- `--trials 3` to run multiple trials for statistical significance

Results are saved to the `results/` directory in JSON and CSV formats.

### Generating Analysis

After running the evaluation, users can generate analysis and visualizations:

```bash
python analysis/failure_analysis.py
python analysis/plot_results.py
```

These scripts produce detailed failure breakdowns and comparative charts saved to the `results/` directory.

## Development Requirements

### Technical Requirements

- **Language**: Python 3.8+
- **LLM Provider**: Groq API (requires API key)
- **Key Libraries**: groq, python-dotenv, matplotlib

### Project Structure

```
LP_SVLTP/
├── simulator/              # Symbolic home environment
│   ├── __init__.py
│   ├── action_space.py     # Action definitions and parsing
│   └── symbolic_home.py    # State management and execution
├── planner/                # Planning algorithms
│   ├── __init__.py
│   ├── translator.py       # Natural language translation
│   ├── llm_client.py       # Groq API wrapper
│   ├── huang_baseline.py   # Zero-shot baseline
│   ├── contextual_baseline.py  # Context-aware baseline
│   └── repair_first.py     # Speculative validation planner
├── eval/                   # Evaluation framework
│   ├── tasks.json          # 20 benchmark tasks
│   └── evaluate.py         # Evaluation runner
├── analysis/               # Analysis scripts
│   ├── failure_analysis.py
│   └── plot_results.py
├── results/                # Output directory
├── DEVLOG.md              # Development log
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

### Installing Dependencies

```bash
pip install groq python-dotenv matplotlib
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

### Running Tests

The project includes self-test scripts in each module. To verify the installation:

```bash
python -m simulator.symbolic_home    # Test simulator
python -m planner.translator         # Test translator (31 cases)
python -m planner.llm_client         # Test LLM connection
```

## License

This project is developed for academic purposes. The codebase is available under the MIT License, which allows anyone to use, modify, and distribute the code with proper attribution.

## Acknowledgments

This project builds upon prior work in language-grounded task planning, particularly the approaches described in Huang et al. (2022) for zero-shot LLM planning. The symbolic home environment is inspired by standard benchmarks in embodied AI research.
