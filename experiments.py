import os
import sys
import subprocess
import itertools
import yaml
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Experiment:
    name: str
    config_overrides: Dict[str, Any]

class ExperimentRunner:
    def __init__(self, base_config_path: str, output_root: str):
        self.base_config_path = base_config_path
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)

    def run_experiment(self, exp: Experiment):
        print(f"--> Starting experiment: {exp.name}")
        exp_dir = os.path.join(self.output_root, exp.name)
        
        # Construct command arguments
        # base config file
        cmd = [sys.executable, 'train.py', self.base_config_path]
        
        # overrides
        for k, v in exp.config_overrides.items():
            cmd.append(f"--{k}={v}")
        
        # ensure output dir is set to experiment dir
        cmd.append(f"--out_dir={exp_dir}")

        # enable tensorboard with specific run name
        cmd.append(f"--tensorboard_run_name={exp.name}")
        
        print(f"    Command: {' '.join(cmd)}")
        
        # Run subprocess
        start_time = time.time()
        try:
            # We wait for it to finish. For parallel, use Popen and poll.
            # Sequential for now to avoid OOM.
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f"    Finished in {duration:.2f}s")
        except subprocess.CalledProcessError as e:
            print(f"    FAILED: {e}")

    def run_grid(self, grid_name: str, base_params: Dict[str, Any], grid_params: Dict[str, List[Any]]):
        """
        Runs a grid search.
        grid_params: {'n_layer': [2, 4], 'n_embd': [128, 256]}
        """
        keys = list(grid_params.keys())
        values = list(grid_params.values())
        
        combinations = list(itertools.product(*values))
        print(f"Scheduled {len(combinations)} experiments for grid: {grid_name}")
        
        for i, combo in enumerate(combinations):
            # Create unique name
            combo_desc = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
            exp_name = f"{grid_name}_{combo_desc}"
            
            # Merge overrides
            overrides = base_params.copy()
            for k, v in zip(keys, combo):
                overrides[k] = v
            
            exp = Experiment(name=exp_name, config_overrides=overrides)
            self.run_experiment(exp)

    def load_metrics(self, grid_name_filter: str = "") -> List[Dict]:
        """
        Loads all metrics.json from the output root, filtering by name.
        """
        results = []
        for d in os.listdir(self.output_root):
            if grid_name_filter and grid_name_filter not in d:
                continue
            
            metrics_path = os.path.join(self.output_root, d, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    data['experiment_name'] = d
                    results.append(data)
        return results

if __name__ == "__main__":
    # Example usage
    # python experiments.py
    # Configure your base experiment here or via CLI?
    # This script is meant to be imported or modified for specific campaigns.
    
    # Example: tiny batch scaling debug
    runner = ExperimentRunner(base_config_path='config/train_shakespeare_char.py', output_root='experiments_out')
    
    # We need a base config file. 
    # Let's assume user has one, or we use a dummy one.
    # For now, just print help.
    print("Import ExperimentRunner from this module to run scaling studies.")
