"""
Visualization of Language Planner Evaluation Results.

Generates plots showing accuracy-latency trade-offs.
"""

import json
import glob
import os

# Use non-interactive backend for saving plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_latest_results():
    """Load the most recent results file."""
    files = glob.glob('results/raw_results_*.json')
    if not files:
        raise FileNotFoundError("No results files found in results/")
    latest = max(files, key=os.path.getctime)
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f)


def compute_summary(data):
    """Compute summary statistics by planner."""
    by_planner = {}
    for r in data:
        planner = r['planner']
        if planner not in by_planner:
            by_planner[planner] = {'success': [], 'api_calls': [], 'elapsed': []}
        by_planner[planner]['success'].append(1 if r['success'] else 0)
        by_planner[planner]['api_calls'].append(r['api_calls'])
        by_planner[planner]['elapsed'].append(r['elapsed_seconds'])
    
    summary = {}
    for planner, metrics in by_planner.items():
        summary[planner] = {
            'success_rate': sum(metrics['success']) / len(metrics['success']),
            'avg_api_calls': sum(metrics['api_calls']) / len(metrics['api_calls']),
            'avg_elapsed': sum(metrics['elapsed']) / len(metrics['elapsed'])
        }
    return summary


def plot_pareto_frontier(summary):
    """Create accuracy-latency trade-off plot."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors and markers for each planner
    styles = {
        'huang': {'color': '#e74c3c', 'marker': 's', 'label': 'Huang (No Context)'},
        'contextual': {'color': '#27ae60', 'marker': '^', 'label': 'Contextual (With Context)'},
        'repair_first': {'color': '#3498db', 'marker': 'o', 'label': 'RepairFirst (Context + Repair)'}
    }
    
    for planner, stats in summary.items():
        style = styles.get(planner, {'color': 'gray', 'marker': 'o', 'label': planner})
        ax.scatter(
            stats['avg_api_calls'], 
            stats['success_rate'] * 100,
            s=300,
            c=style['color'],
            marker=style['marker'],
            label=style['label'],
            edgecolors='white',
            linewidths=2,
            zorder=10
        )
        # Add annotation
        ax.annotate(
            f"{stats['success_rate']*100:.0f}%",
            (stats['avg_api_calls'], stats['success_rate'] * 100),
            textcoords="offset points",
            xytext=(0, 15),
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_xlabel('Average API Calls (Latency Proxy)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Accuracy-Latency Trade-off\n(Language Planner Comparison)', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0.8, 1.4)
    ax.set_ylim(30, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # Add Pareto frontier line
    pareto_x = [1.0, 1.0]  # Contextual is Pareto optimal
    pareto_y = [85, 85]
    ax.axhline(y=85, color='green', linestyle='--', alpha=0.3, label='Pareto Frontier')
    
    plt.tight_layout()
    plt.savefig('results/pareto.png', dpi=150, bbox_inches='tight')
    print("Saved: results/pareto.png")
    
    return fig


def plot_by_difficulty(data):
    """Create bar chart comparing success by difficulty."""
    
    # Group by planner and difficulty
    stats = {}
    for r in data:
        key = (r['planner'], r['difficulty'])
        if key not in stats:
            stats[key] = {'success': 0, 'total': 0}
        stats[key]['total'] += 1
        if r['success']:
            stats[key]['success'] += 1
    
    # Prepare data
    planners = ['huang', 'contextual', 'repair_first']
    difficulties = ['easy', 'medium', 'hard']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(difficulties))
    width = 0.25
    
    colors = {'huang': '#e74c3c', 'contextual': '#27ae60', 'repair_first': '#3498db'}
    labels = {'huang': 'Huang', 'contextual': 'Contextual', 'repair_first': 'RepairFirst'}
    
    for i, planner in enumerate(planners):
        rates = []
        for diff in difficulties:
            key = (planner, diff)
            if key in stats:
                rates.append(100 * stats[key]['success'] / stats[key]['total'])
            else:
                rates.append(0)
        
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], rates, width, 
                      label=labels[planner], color=colors[planner], alpha=0.85)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.annotate(f'{rate:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Task Difficulty', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate by Difficulty Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/by_difficulty.png', dpi=150, bbox_inches='tight')
    print("Saved: results/by_difficulty.png")
    
    return fig


if __name__ == "__main__":
    data = load_latest_results()
    summary = compute_summary(data)
    
    print("\n=== Summary ===")
    for planner, stats in summary.items():
        print(f"{planner}: {stats['success_rate']*100:.1f}% success, {stats['avg_api_calls']:.2f} API calls")
    
    print("\n=== Generating Plots ===")
    plot_pareto_frontier(summary)
    plot_by_difficulty(data)
    
    print("\nDone!")
