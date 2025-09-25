# Curriculum Learning Configurations for Curriculum-Augmented GFlowNets for mRNA Design

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
