from .utils import Config
from .experiment import Experiment

def experiment():
    p = Config.argparser()
    args = p.parse_args()
    print("Parsed arguments:", args)

    config = Config(**vars(args))
    exp = Experiment(config)
    result = exp.run()
    print(f"Experiment result: {result}")