"""Module with base class for environment components."""

import logging


class EnvComponent:
    """Base class for environment component."""

    def __init__(self) -> None:
        """Base class for environment component."""

        # Setting log with handler, that allows access to log
        # via self.log_handler.show_logs()
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.info("Environment component initialised.")

    def set_log_level(self, level: str) -> None:
        """Set level of logger.

        Args:
            level (str): level to set logger to.
        """
        self.logger.setLevel(level)

    def set_log_handler(self, handler: logging.Handler) -> None:
        """Set handler for logger.

        Args:
            handler (logging.Handler): handler to use
        """

        self.logger.addHandler(handler)
