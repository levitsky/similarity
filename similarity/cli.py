from .experiment import Experiment, Config

def experiment():
    config = Config()  # Placeholder for actual configuration
    exp = Experiment(config)
    result = exp.run()
    print(f"Experiment result: {result}")