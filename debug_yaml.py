
import yaml
import sys

yaml_content = "learning_rate: 6e-4"
data = yaml.safe_load(yaml_content)
print(f"Content: {yaml_content}")
print(f"Parsed: {data['learning_rate']}")
print(f"Type: {type(data['learning_rate'])}")

with open('config/train_full.yaml', 'r') as f:
    full_data = yaml.safe_load(f)
    lr = full_data.get('learning_rate')
    print(f"File LR: {lr}")
    print(f"File LR Type: {type(lr)}")
