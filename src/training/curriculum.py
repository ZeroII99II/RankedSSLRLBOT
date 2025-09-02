"""
YAML-driven curriculum system for SSL bot training.
Manages progression from Bronze to SSL with phase-specific configurations
for rewards, scenarios, and evaluation metrics.
"""

import yaml
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumPhase:
    """Configuration for a single curriculum phase."""
    name: str
    description: str
    
    # Training configuration
    reward_weights: Dict[str, float]
    scenario_weights: Dict[str, float]
    opponent_mix: Dict[str, float]  # self-play, scripted, etc.
    
    # Progression criteria
    progression_gates: Dict[str, Any]
    min_training_steps: int
    max_training_steps: int
    
    # Evaluation metrics
    eval_metrics: Dict[str, float]  # target values for progression


class CurriculumManager:
    """
    Manages curriculum progression from Bronze to SSL.
    
    Phases:
    - Bronze/Silver: Basic movement, touches, recoveries
    - Gold/Plat: Boost management, positioning, basic aerials  
    - Diamond/Champ: Advanced positioning, power shots, dribbles
    - GC/SSL: Aerial mechanics, wall reads, double taps, flip resets
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.phases: Dict[str, CurriculumPhase] = {}
        self.current_phase: str = "bronze"
        self.phase_history: List[Tuple[str, int]] = []  # (phase, steps)
        self.training_steps: int = 0
        self.games_played: int = 0
        
        self._load_curriculum_config()
    
    def _load_curriculum_config(self):
        """Load curriculum configuration from YAML file."""
        if not self.config_path.exists():
            self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for phase_name, phase_config in config['phases'].items():
            self.phases[phase_name] = CurriculumPhase(
                name=phase_name,
                description=phase_config['description'],
                reward_weights=phase_config['reward_weights'],
                scenario_weights=phase_config['scenario_weights'],
                opponent_mix=phase_config['opponent_mix'],
                progression_gates=phase_config['progression_gates'],
                min_training_steps=phase_config['min_training_steps'],
                max_training_steps=phase_config['max_training_steps'],
                eval_metrics=phase_config['eval_metrics']
            )
    
    def _create_default_config(self):
        """Create default curriculum configuration."""
        default_config = {
            'phases': {
                'bronze': {
                    'description': 'Basic movement, touches, recoveries',
                    'reward_weights': {
                        'ball_to_goal_velocity': 0.3,
                        'touch_quality': 0.5,
                        'recovery': 1.0,
                        'boost_economy': 0.5,
                        'aerial_intercept': 0.1,
                        'backboard_read': 0.1,
                        'double_tap_setup': 0.1,
                        'flip_reset': 0.1,
                        'fast_aerial': 0.1
                    },
                    'scenario_weights': {
                        'kickoff': 0.3,
                        'ground_play': 0.4,
                        'simple_aerial': 0.1,
                        'defensive_clear': 0.1,
                        'boost_collection': 0.1
                    },
                    'opponent_mix': {
                        'self_play': 0.8,
                        'scripted_basic': 0.2
                    },
                    'progression_gates': {
                        'min_win_rate': 0.6,
                        'min_goal_rate': 0.3,
                        'min_touch_rate': 0.4
                    },
                    'min_training_steps': 100000,
                    'max_training_steps': 500000,
                    'eval_metrics': {
                        'win_rate': 0.6,
                        'goal_rate': 0.3,
                        'touch_rate': 0.4,
                        'recovery_time': 2.0
                    }
                },
                'silver': {
                    'description': 'Improved movement, basic aerials, wall bounces',
                    'reward_weights': {
                        'ball_to_goal_velocity': 0.5,
                        'touch_quality': 0.7,
                        'recovery': 1.0,
                        'boost_economy': 0.7,
                        'aerial_intercept': 0.2,
                        'backboard_read': 0.2,
                        'double_tap_setup': 0.1,
                        'flip_reset': 0.1,
                        'fast_aerial': 0.2
                    },
                    'scenario_weights': {
                        'kickoff': 0.25,
                        'ground_play': 0.3,
                        'simple_aerial': 0.2,
                        'wall_bounce': 0.1,
                        'defensive_clear': 0.1,
                        'boost_collection': 0.05
                    },
                    'opponent_mix': {
                        'self_play': 0.7,
                        'scripted_basic': 0.2,
                        'scripted_intermediate': 0.1
                    },
                    'progression_gates': {
                        'min_win_rate': 0.65,
                        'min_goal_rate': 0.35,
                        'min_aerial_success': 0.2
                    },
                    'min_training_steps': 200000,
                    'max_training_steps': 800000,
                    'eval_metrics': {
                        'win_rate': 0.65,
                        'goal_rate': 0.35,
                        'aerial_success': 0.2,
                        'wall_read_success': 0.1
                    }
                },
                'gold': {
                    'description': 'Boost management, positioning, wall reads',
                    'reward_weights': {
                        'ball_to_goal_velocity': 0.7,
                        'touch_quality': 1.0,
                        'recovery': 1.0,
                        'boost_economy': 1.0,
                        'aerial_intercept': 0.4,
                        'backboard_read': 0.3,
                        'double_tap_setup': 0.2,
                        'flip_reset': 0.1,
                        'fast_aerial': 0.4
                    },
                    'scenario_weights': {
                        'kickoff': 0.2,
                        'ground_play': 0.25,
                        'simple_aerial': 0.2,
                        'wall_bounce': 0.15,
                        'backboard_clear': 0.1,
                        'defensive_clear': 0.05,
                        'boost_collection': 0.05
                    },
                    'opponent_mix': {
                        'self_play': 0.6,
                        'scripted_basic': 0.2,
                        'scripted_intermediate': 0.2
                    },
                    'progression_gates': {
                        'min_win_rate': 0.7,
                        'min_goal_rate': 0.4,
                        'min_boost_efficiency': 0.6
                    },
                    'min_training_steps': 300000,
                    'max_training_steps': 1000000,
                    'eval_metrics': {
                        'win_rate': 0.7,
                        'goal_rate': 0.4,
                        'boost_efficiency': 0.6,
                        'wall_read_success': 0.2
                    }
                },
                'platinum': {
                    'description': 'Advanced positioning, backboard plays',
                    'reward_weights': {
                        'ball_to_goal_velocity': 0.8,
                        'touch_quality': 1.0,
                        'recovery': 1.0,
                        'boost_economy': 1.0,
                        'aerial_intercept': 0.6,
                        'backboard_read': 0.5,
                        'double_tap_setup': 0.3,
                        'flip_reset': 0.2,
                        'fast_aerial': 0.6
                    },
                    'scenario_weights': {
                        'kickoff': 0.15,
                        'ground_play': 0.2,
                        'simple_aerial': 0.2,
                        'wall_bounce': 0.15,
                        'backboard_clear': 0.15,
                        'aerial_intercept': 0.1,
                        'defensive_clear': 0.05
                    },
                    'opponent_mix': {
                        'self_play': 0.5,
                        'scripted_intermediate': 0.3,
                        'scripted_advanced': 0.2
                    },
                    'progression_gates': {
                        'min_win_rate': 0.75,
                        'min_goal_rate': 0.45,
                        'min_backboard_success': 0.3
                    },
                    'min_training_steps': 400000,
                    'max_training_steps': 1200000,
                    'eval_metrics': {
                        'win_rate': 0.75,
                        'goal_rate': 0.45,
                        'backboard_success': 0.3,
                        'aerial_intercept_success': 0.25
                    }
                },
                'diamond': {
                    'description': 'Power shots, dribbles, advanced aerials',
                    'reward_weights': {
                        'ball_to_goal_velocity': 1.0,
                        'touch_quality': 1.0,
                        'recovery': 1.0,
                        'boost_economy': 1.0,
                        'aerial_intercept': 0.8,
                        'backboard_read': 0.7,
                        'double_tap_setup': 0.5,
                        'flip_reset': 0.3,
                        'fast_aerial': 0.8
                    },
                    'scenario_weights': {
                        'kickoff': 0.1,
                        'ground_play': 0.15,
                        'simple_aerial': 0.15,
                        'wall_bounce': 0.15,
                        'backboard_clear': 0.15,
                        'aerial_intercept': 0.15,
                        'double_tap_setup': 0.1,
                        'defensive_clear': 0.05
                    },
                    'opponent_mix': {
                        'self_play': 0.4,
                        'scripted_intermediate': 0.3,
                        'scripted_advanced': 0.3
                    },
                    'progression_gates': {
                        'min_win_rate': 0.8,
                        'min_goal_rate': 0.5,
                        'min_double_tap_success': 0.2
                    },
                    'min_training_steps': 500000,
                    'max_training_steps': 1500000,
                    'eval_metrics': {
                        'win_rate': 0.8,
                        'goal_rate': 0.5,
                        'double_tap_success': 0.2,
                        'power_shot_success': 0.3
                    }
                },
                'champion': {
                    'description': 'Advanced aerials, flip resets, complex plays',
                    'reward_weights': {
                        'ball_to_goal_velocity': 1.0,
                        'touch_quality': 1.0,
                        'recovery': 1.0,
                        'boost_economy': 1.0,
                        'aerial_intercept': 1.0,
                        'backboard_read': 0.9,
                        'double_tap_setup': 0.7,
                        'flip_reset': 0.5,
                        'fast_aerial': 1.0
                    },
                    'scenario_weights': {
                        'kickoff': 0.08,
                        'ground_play': 0.12,
                        'simple_aerial': 0.12,
                        'wall_bounce': 0.15,
                        'backboard_clear': 0.15,
                        'aerial_intercept': 0.15,
                        'double_tap_setup': 0.15,
                        'flip_reset_setup': 0.08
                    },
                    'opponent_mix': {
                        'self_play': 0.3,
                        'scripted_advanced': 0.4,
                        'scripted_expert': 0.3
                    },
                    'progression_gates': {
                        'min_win_rate': 0.85,
                        'min_goal_rate': 0.55,
                        'min_flip_reset_success': 0.1
                    },
                    'min_training_steps': 600000,
                    'max_training_steps': 1800000,
                    'eval_metrics': {
                        'win_rate': 0.85,
                        'goal_rate': 0.55,
                        'flip_reset_success': 0.1,
                        'complex_aerial_success': 0.2
                    }
                },
                'gc': {
                    'description': 'Expert aerial mechanics, ceiling plays',
                    'reward_weights': {
                        'ball_to_goal_velocity': 1.0,
                        'touch_quality': 1.0,
                        'recovery': 1.0,
                        'boost_economy': 1.0,
                        'aerial_intercept': 1.0,
                        'backboard_read': 1.0,
                        'double_tap_setup': 0.9,
                        'flip_reset': 0.7,
                        'fast_aerial': 1.0
                    },
                    'scenario_weights': {
                        'kickoff': 0.05,
                        'ground_play': 0.1,
                        'simple_aerial': 0.1,
                        'wall_bounce': 0.15,
                        'backboard_clear': 0.15,
                        'aerial_intercept': 0.15,
                        'double_tap_setup': 0.15,
                        'flip_reset_setup': 0.1,
                        'ceiling_pinch': 0.05
                    },
                    'opponent_mix': {
                        'self_play': 0.2,
                        'scripted_advanced': 0.3,
                        'scripted_expert': 0.5
                    },
                    'progression_gates': {
                        'min_win_rate': 0.9,
                        'min_goal_rate': 0.6,
                        'min_ceiling_success': 0.15
                    },
                    'min_training_steps': 700000,
                    'max_training_steps': 2000000,
                    'eval_metrics': {
                        'win_rate': 0.9,
                        'goal_rate': 0.6,
                        'ceiling_success': 0.15,
                        'expert_aerial_success': 0.25
                    }
                },
                'ssl': {
                    'description': 'SSL-level mechanics, all advanced plays',
                    'reward_weights': {
                        'ball_to_goal_velocity': 1.0,
                        'touch_quality': 1.0,
                        'recovery': 1.0,
                        'boost_economy': 1.0,
                        'aerial_intercept': 1.0,
                        'backboard_read': 1.0,
                        'double_tap_setup': 1.0,
                        'flip_reset': 1.0,
                        'fast_aerial': 1.0
                    },
                    'scenario_weights': {
                        'kickoff': 0.05,
                        'ground_play': 0.08,
                        'simple_aerial': 0.08,
                        'wall_bounce': 0.12,
                        'backboard_clear': 0.15,
                        'aerial_intercept': 0.15,
                        'double_tap_setup': 0.15,
                        'flip_reset_setup': 0.12,
                        'ceiling_pinch': 0.1
                    },
                    'opponent_mix': {
                        'self_play': 0.1,
                        'scripted_expert': 0.9
                    },
                    'progression_gates': {
                        'min_win_rate': 0.95,
                        'min_goal_rate': 0.65,
                        'min_ssl_mechanics': 0.3
                    },
                    'min_training_steps': 800000,
                    'max_training_steps': 2500000,
                    'eval_metrics': {
                        'win_rate': 0.95,
                        'goal_rate': 0.65,
                        'ssl_mechanics_success': 0.3,
                        'overall_skill': 0.9
                    }
                }
            }
        }
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    def get_current_phase(self) -> CurriculumPhase:
        """Get current curriculum phase."""
        return self.phases[self.current_phase]
    
    def get_phase_order(self) -> List[str]:
        """Get ordered list of curriculum phases."""
        return ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'champion', 'gc', 'ssl']
    
    def can_progress(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if current phase progression criteria are met."""
        current_phase = self.get_current_phase()
        
        # Check minimum training steps
        if self.training_steps < current_phase.min_training_steps:
            return False

        gates = current_phase.progression_gates

        # Check evaluation metrics first to ensure missing metrics are logged
        missing_metrics = []
        for metric, threshold in gates.items():
            if metric == "min_games":
                continue
            value = eval_metrics.get(metric)
            if value is None:
                logger.warning("Missing evaluation metric: %s", metric)
                missing_metrics.append(metric)
                continue
            if value < threshold:
                return False

        if missing_metrics:
            return False

        # Finally check overall training statistics like games played
        min_games = gates.get("min_games", 0)
        if self.games_played < min_games:
            return False

        return True
    
    def should_progress(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if should progress to next phase."""
        if not self.can_progress(eval_metrics):
            return False
        
        current_phase = self.get_current_phase()
        
        # Check if max training steps reached
        if self.training_steps >= current_phase.max_training_steps:
            return True
        
        # Check if all metrics exceed targets significantly
        for metric, target in current_phase.eval_metrics.items():
            if metric in eval_metrics:
                if eval_metrics[metric] < target * 1.1:  # 10% above target
                    return False
            else:
                return False
        
        return True
    
    def progress_to_next_phase(self) -> bool:
        """Progress to next curriculum phase."""
        phase_order = self.get_phase_order()
        current_index = phase_order.index(self.current_phase)
        
        if current_index < len(phase_order) - 1:
            # Record phase completion
            self.phase_history.append((self.current_phase, self.training_steps))
            
            # Move to next phase
            self.current_phase = phase_order[current_index + 1]
            self.training_steps = 0  # Reset step counter
            self.games_played = 0
            
            print(f"Progressed to {self.current_phase} phase")
            return True
        
        return False  # Already at final phase
    
    def update_training_steps(self, steps: int):
        """Update training step counter."""
        self.training_steps += steps

    def update_games(self, games: int):
        """Update completed games counter."""

        """Update games played counter."""
        self.games_played += games
    
    def get_reward_weights(self) -> Dict[str, float]:
        """Get current phase reward weights."""
        return self.get_current_phase().reward_weights
    
    def get_scenario_weights(self) -> Dict[str, float]:
        """Get current phase scenario weights."""
        return self.get_current_phase().scenario_weights
    
    def get_opponent_mix(self) -> Dict[str, float]:
        """Get current phase opponent mix."""
        return self.get_current_phase().opponent_mix
    
    def get_eval_metrics(self) -> Dict[str, float]:
        """Get current phase evaluation metrics."""
        return self.get_current_phase().eval_metrics
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """Get current phase progress information."""
        current_phase = self.get_current_phase()
        phase_order = self.get_phase_order()
        current_index = phase_order.index(self.current_phase)
        
        return {
            'current_phase': self.current_phase,
            'phase_index': current_index,
            'total_phases': len(phase_order),
            'training_steps': self.training_steps,
            'games_played': self.games_played,
            'min_steps': current_phase.min_training_steps,
            'max_steps': current_phase.max_training_steps,
            'min_games': current_phase.progression_gates.get('min_games', 0),
            'phase_history': self.phase_history,
            'description': current_phase.description
        }
    
    def save_progress(self, save_path: str):
        """Save curriculum progress to file."""
        progress_data = {
            'current_phase': self.current_phase,
            'training_steps': self.training_steps,
            'games_played': self.games_played,
            'phase_history': self.phase_history
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(progress_data, f, default_flow_style=False, indent=2)
    
    def load_progress(self, save_path: str):
        """Load curriculum progress from file."""
        if Path(save_path).exists():
            with open(save_path, 'r') as f:
                progress_data = yaml.safe_load(f)
            
            self.current_phase = progress_data.get('current_phase', 'bronze')
            self.training_steps = progress_data.get('training_steps', 0)
            self.games_played = progress_data.get('games_played', 0)
            self.phase_history = progress_data.get('phase_history', [])
