import yaml

# Load bc_exp_cfg.yaml
with open("bc_exp_cfg.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  
print(config['base'])