import logging
import os
import sys

import cv2
import yaml

PROGRESS = 25


class MyLogger(logging.Logger):
    PROGRESS = PROGRESS
    LOG_FORMATTER = '%(asctime)s - %(levelname)-10s - %(message)s'
    DEF_LOGGING_LEVEL = logging.WARNING

    LOG_FOLDER = "logs/"
    LOG_IMG_FOLDER = os.path.join(LOG_FOLDER, "img")
    ABSOLUTE_FOLDER = ""

    def __init__(self, log_name, level=None):
        logging.Logger.__init__(self, log_name)
        self.formatter = logging.Formatter(self.LOG_FORMATTER)
        self.initLogger(level)

    def initLogger(self, level=None):
        self.setLevel(level or self.DEF_LOGGING_LEVEL)
        self.propagate = False

    def add_handler(self, log_file, use_syslog):
        if use_syslog:
            handler = logging.handlers.SysLogHandler(address='/dev/log')
        elif log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(self.formatter)
        self.addHandler(handler)
        return handler

    def addHandlers(self, log_file=None, progress_file=None, use_syslog=False):
        self.logger_hdlr = self.add_handler(log_file, use_syslog)
        if progress_file:
            self.progress_hdlr = self.add_handler(progress_file, use_syslog)
            self.progress_hdlr.setLevel(self.PROGRESS)
        else:
            self.progress_hdlr = None

    def progress(self, image_name, stage_name, image, *args):
        file_name = image_name + "_" + stage_name + ".png"
        cv2.imwrite(os.path.join(self.LOG_IMG_FOLDER, file_name), image)
        msg = f"Image {file_name} saved"
        self.log(self.PROGRESS, msg)

    def display_image(self, image_name, stage_name, image, *args, **kwargs):
        if self.level == logging.DEBUG:
            file_name = image_name + "_" + stage_name
            cv2.imshow(file_name, image)
            cv2.waitKey(0)
            msg = f"Image {file_name} displayed"
            return super(MyLogger, self).debug(msg, *args, **kwargs)

    def log_image(self, image_name, stage_name, image, save=False, *args, **kwargs):
        if save:
            self.progress(image_name, stage_name, image, *args)
        self.display_image(image_name, stage_name, image, *args, **kwargs)

    def log_plot(self, plt, image_name, stage_name, save=True):
        if self.level == logging.DEBUG:
            plt.show()
        if save:
            file_name = image_name + "_" + stage_name + ".png"
            plt.savefig(str(os.path.join(self.LOG_IMG_FOLDER, file_name)))

    def log_figure(self, fig, image_name, save=True):
        if save:
            file_name = image_name + ".png"
            fig.savefig(str(os.path.join(self.LOG_IMG_FOLDER, file_name)))

    def load_level(self, settings='settings.yaml'):
        ABSOLUTE_FOLDER = os.path.join(os.path.abspath(os.path.join("SRC", os.pardir)))
        settings_file = os.path.join(ABSOLUTE_FOLDER, settings)
        with open(settings_file) as info:
            settings_dict = yaml.safe_load(info)
        self.level = dict(settings_dict)['LOGGING_LEVEL']
        self.setLevel(self.level)


def setup_logger(progress_file=None):
    logging.setLoggerClass(MyLogger)
    logging.addLevelName(PROGRESS, 'PROGRESS')
    logger = logging.getLogger(__name__)
    logger.addHandlers(progress_file)
    logger.load_level()
    return logger
