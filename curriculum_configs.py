# Curriculum Learning Configurations for mRNA Design Research
# These configurations highlight different aspects of curriculum learning effectiveness

def get_curriculum_configs():
    """
    Returns three different curriculum learning configurations designed to showcase
    different aspects of curriculum learning effectiveness in mRNA design tasks.

    These configurations are designed for research paper evaluation to demonstrate:
    1. Conservative vs Aggressive learning strategies
    2. Different progress estimation methods
    3. Various exploration-exploitation trade-offs
    """

    configs = {

        # Configuration 1: Conservative Learning Progress (EMA-based)
        # This configuration emphasizes stability and gradual progress
        "conservative_ema": {
            'name': 'Conservative EMA-based Curriculum',
            'description': 'Uses exponential moving average for stable progress estimation with moderate exploration',
            'lpe': 'Online',              # Simple online learning progress (most conservative)
            'acp': 'LP',                  # Learning Progress attention (simpler than MR)
            'a2d': 'GreedyProp',          # Proportional with exploration
            'a2d_eps': 0.15,              # Higher exploration (15% uniform)
            'lpe_alpha': 0.05,            # Very slow EMA updates (conservative)
            'acp_MR_K': 15,               # Smaller window for responsiveness
            'acp_MR_power': 2,            # Lower power (less aggressive task selection)
            'acp_MR_pot_prop': 0.4,       # Lower potential emphasis
            'acp_MR_att_pred': 0.1,       # Lower predecessor attention
            'acp_MR_att_succ': 0.05,      # Lower successor attention
        },

        # Configuration 2: Aggressive Exploration (Sampling-based)
        # This configuration emphasizes exploration and rapid adaptation
        "aggressive_sampling": {
            'name': 'Aggressive Sampling-based Curriculum',
            'description': 'Uses sampling-based progress estimation with high exploration for rapid adaptation',
            'lpe': 'Sampling',            # Sampling-based progress (most adaptive)
            'acp': 'MR',                  # Mastering Rate attention (most sophisticated)
            'a2d': 'Boltzmann',           # Boltzmann distribution for smooth exploration
            'a2d_tau': 0.5,               # Higher temperature (more exploration)
            'lpe_K': 10,                  # Smaller window for faster adaptation
            'acp_MR_K': 10,               # Smaller window for faster updates
            'acp_MR_power': 8,            # Higher power (more aggressive task selection)
            'acp_MR_pot_prop': 0.8,       # Higher potential emphasis (focus on promising tasks)
            'acp_MR_att_pred': 0.4,       # Higher predecessor attention (stability)
            'acp_MR_att_succ': 0.2,       # Higher successor attention (forward push)
        },

        # Configuration 3: Balanced Multi-Objective (Prop-based)
        # This configuration balances exploration and exploitation for multi-objective optimization
        "balanced_prop": {
            'name': 'Balanced Proportional Curriculum',
            'description': 'Uses linear regression progress estimation with proportional task selection for balanced learning',
            'lpe': 'Linreg',              # Linear regression (balanced, good for noisy rewards)
            'acp': 'MR',                  # Mastering Rate attention
            'a2d': 'Prop',                # Pure proportional (no additional exploration)
            'a2d_eps': 0.0,               # No additional uniform exploration
            'lpe_K': 25,                  # Larger window for stability
            'acp_MR_K': 25,               # Larger window for smooth updates
            'acp_MR_power': 4,            # Moderate power (balanced task selection)
            'acp_MR_pot_prop': 0.6,       # Moderate potential emphasis
            'acp_MR_att_pred': 0.3,       # Moderate predecessor attention
            'acp_MR_att_succ': 0.1,       # Moderate successor attention
        }
    }

    return configs

def get_configuration_comparison():
    """
    Returns a comparison table of the three configurations highlighting their key differences
    and expected behaviors for research paper analysis.
    """

    comparison = {
        'aspects': [
            'Learning Progress Estimation',
            'Attention Computation',
            'Distribution Mapping',
            'Exploration Strategy',
            'Adaptation Speed',
            'Stability',
            'Best Use Case'
        ],
        'conservative_ema': [
            'Online EMA (α=0.05)',
            'Learning Progress only',
            'GreedyProp (ε=0.15)',
            'Moderate exploration',
            'Slow adaptation',
            'High stability',
            'Stable environments, gradual learning'
        ],
        'aggressive_sampling': [
            'Sampling-based (K=10)',
            'Mastering Rate (power=8)',
            'Boltzmann (τ=0.5)',
            'High exploration',
            'Fast adaptation',
            'Lower stability',
            'Dynamic environments, rapid learning'
        ],
        'balanced_prop': [
            'Linear Regression (K=25)',
            'Mastering Rate (power=4)',
            'Proportional (ε=0.0)',
            'Balanced exploration',
            'Moderate adaptation',
            'Moderate stability',
            'Multi-objective optimization'
        ]
    }

    return comparison

def get_research_insights():
    """
    Returns research insights and hypotheses for each configuration that can be
    used in the research paper to justify the experimental design.
    """

    insights = {
        'conservative_ema': {
            'hypothesis': 'Conservative EMA-based curriculum will show stable but slower convergence, suitable for environments with high reward noise',
            'expected_behavior': [
                'Gradual task progression with minimal oscillation',
                'Lower variance in task selection over time',
                'Better performance on longer sequences due to stable learning',
                'May struggle with rapid adaptation to new task difficulties'
            ],
            'research_value': 'Demonstrates the importance of stability in curriculum learning for complex biological optimization tasks'
        },

        'aggressive_sampling': {
            'hypothesis': 'Aggressive sampling-based curriculum will show rapid adaptation but higher variance, suitable for dynamic optimization landscapes',
            'expected_behavior': [
                'Rapid task switching based on immediate progress signals',
                'Higher exploration of task space',
                'Better performance on shorter sequences due to quick adaptation',
                'May show instability in task selection patterns'
            ],
            'research_value': 'Highlights the trade-off between exploration and exploitation in curriculum learning for mRNA design'
        },

        'balanced_prop': {
            'hypothesis': 'Balanced proportional curriculum will achieve optimal trade-off between stability and adaptation for multi-objective mRNA optimization',
            'expected_behavior': [
                'Smooth task progression with moderate exploration',
                'Balanced attention to all objectives (GC, MFE, CAI)',
                'Consistent performance across different sequence lengths',
                'Stable convergence with reasonable adaptation speed'
            ],
            'research_value': 'Shows the effectiveness of balanced curriculum learning for complex multi-objective biological optimization problems'
        }
    }

    return insights

if __name__ == "__main__":
    # Example usage and demonstration
    configs = get_curriculum_configs()
    comparison = get_configuration_comparison()
    insights = get_research_insights()

    print("Curriculum Learning Configurations for mRNA Design Research")
    print("=" * 60)

    for config_name, config in configs.items():
        print(f"\n{config['name']}")
        print(f"Description: {config['description']}")
        print("Key Parameters:")
        for key, value in config.items():
            if key not in ['name', 'description']:
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Configuration Comparison")
    print("=" * 60)

    # Print comparison table
    aspects = comparison['aspects']
    for i, aspect in enumerate(aspects):
        print(f"\n{aspect}:")
        print(f"  Conservative EMA: {comparison['conservative_ema'][i]}")
        print(f"  Aggressive Sampling: {comparison['aggressive_sampling'][i]}")
        print(f"  Balanced Prop: {comparison['balanced_prop'][i]}")

    print("\n" + "=" * 60)
    print("Research Insights")
    print("=" * 60)

    for config_name, insight in insights.items():
        print(f"\n{configs[config_name]['name']}:")
        print(f"Hypothesis: {insight['hypothesis']}")
        print("Expected Behaviors:")
        for behavior in insight['expected_behavior']:
            print(f"  - {behavior}")
        print(f"Research Value: {insight['research_value']}")
