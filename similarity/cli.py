from .utils import Config
from .experiment import Experiment
import logging


def experiment():
    p = Config.argparser()
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Parsed arguments: %s", args)

    config = Config(**vars(args))
    exp = Experiment(config)
    result = exp.run()
    logger.debug("Experiment result: %s", result)
