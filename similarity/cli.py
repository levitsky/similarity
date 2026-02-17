from .utils import Config
from .experiment import Experiment
import logging


def experiment():
    p = Config.argparser()
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Parsed arguments: %s", args)
    kw = vars(args)
    kw.pop("verbose", None)  # Remove verbose from config kwargs

    config = Config(**kw)
    exp = Experiment(config)
    result = exp.run()
    logger.info("Experiment result: %s", result)


if __name__ == "__main__":
    experiment()
