
import pandas as pd
import numpy as np
import yaml
import os

def generate_data(num_samples: int, output_path: str):
    np.random.seed(42)
    feature1 = np.random.normal(50, 15, num_samples)
    feature2 = np.random.uniform(0, 100, num_samples)
    feature3 = np.random.poisson(3, num_samples)
    risk = (feature1 / 50) + (feature2 / 100) - (feature3 / 5)
    target = np.random.binomial(1, 1 / (1 + np.exp(-risk)))
    df = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'feature3': feature3, 'target': target})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated data at {output_path}")

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_data(config['data']['num_samples'], config['data']['raw_path'])
