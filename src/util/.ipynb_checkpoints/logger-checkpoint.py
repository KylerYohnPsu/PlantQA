import logging
import threading
import colorlog
import sys

# The colors of each log type
# TODO: move to config.py
LOG_COLORS= {
	"DEBUG": "cyan",
	"INFO": "black",
	"WARNING": "yellow",
	"ERROR": "red",
	"CRITICAL": "bold_red",
}

class Logger:
	"""
	Class used for logging.
	This is essentially a static class, meaning you don't have to instantiate it to use it
	You just import it, then call Logger.info/debug/etc.
	"""
	
	# The thing that does the logging
	_logger= None
	
	# Lock to make sure this class can be used by any thread at any time
	_lock= threading.Lock()
	
	#@classmethod is what allows this class to function as a static class
	@classmethod
	def _initialize(log_class, name="logger", level=logging.DEBUG):
		if log_class._logger is not None:
			return # already initalized
			
		with log_class._lock:
			logger= colorlog.getLogger(name)
			logger.setLevel(level)

            # we have to specify stdout, otherwise jupyter puts a big dumb red box around the logs
			handler= colorlog.StreamHandler(stream=sys.stdout)
			
			# Format all logs
			formatter= colorlog.ColoredFormatter(
				"%(log_color)s%(asctime)s [%(levelname)-8s]: %(message)s%(reset)s",
				datefmt="%H:%M:%S",
				log_colors= LOG_COLORS
			)
			
			handler.setFormatter(formatter)
			
			logger.addHandler(handler)
			logger.propagate= False # helps reuduce duplicate logs appearing
			
			log_class._logger= logger
			# Logger is now initialized
			
	@classmethod
	def get(log_class):
		if log_class._logger is None:
			log_class._initialize()
		return log_class._logger
		
	#### THESE ARE THE FUNCTIONS YOU CALL TO LOG ######
	
	@classmethod
	def debug(log_class, message, *args, **kwargs):
		logger= log_class.get()
		logger.debug(message, *args, **kwargs)
		
	@classmethod
	def info(log_class, message, *args, **kwargs):
		logger= log_class.get()
		logger.info(message, *args, **kwargs)
		
	@classmethod
	def warning(log_class, message, *args, **kwargs):
		logger= log_class.get()
		logger.warning(message, *args, **kwargs)
		
	@classmethod
	def error(log_class, message, *args, **kwargs):
		logger= log_class.get()
		logger.error(message, *args, **kwargs)
		
	@classmethod
	def critical(log_class, message, *args, **kwargs):
		logger= log_class.get()
		logger.critical(message, *args, **kwargs)
		
			
			
