"""Module with utils for logging."""

import logging

import IPython.display
import ipywidgets as widgets


# Logging widgets
class OutputWidgetHandler(logging.Handler):
    """Logging handler sending logs to an output widget.

    This version is copied (with minor adaptations) from:
    https://github.com/ai4er-cdt/gtc-biodiversity/blob/main/geograph/visualisation/widget_utils.py
    Originally taken from:
    https://ipywidgets.readthedocs.io/en/latest/examples/Output%20Widget.html
    """

    def __init__(self, *args, max_len=30, **kwargs):
        """Logging handler sending logs to an output widget.

        Args:
            max_len (int, optional): maximum length of log in lines. Defaults to 30.
        """
        super().__init__(*args, **kwargs)
        self.max_len = max_len
        layout = {
            "width": "100%",
            "border": "1px solid black",
        }
        self.out = widgets.Output(layout=layout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.setFormatter(formatter)

    def emit(self, record):
        """Overload of logging.Handler method."""
        formatted_record = self.format(record)
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        if len(self.out.outputs) > self.max_len:
            self.out.outputs = self.out.outputs[1:] + (new_output,)
        else:
            self.out.outputs = self.out.outputs + (new_output,)

    def show_logs(self):
        """Show the logs."""
        IPython.display.display(self.out)

    def clear_logs(self):
        """Clear the current logs."""
        self.out.clear_output()
