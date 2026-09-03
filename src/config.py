import logging

####### Logging Configuration ########

# The lowest level logs to show
# Options: [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
LOG_LEVEL= logging.DEBUG

# The format of logging outputs
LOG_PATTERN= "%(log_color)s%(asctime)s [%(levelname)-8s]: %(message)s%(reset)s"

# The color of each log type
LOG_COLORS= {
	"DEBUG": "cyan",
	"INFO": "black",
	"WARNING": "yellow",
	"ERROR": "red",
	"CRITICAL": "bold_red",
}

