# This python module will be used to monitor the progress of file creations
# We do this in 2 modes, by either monitoring a external log file, or by
# directly monitoring a folder for creation of new files.
# We will ask the user to supply the path to the log file or the folder to monitor
# We will monitor and calculate estimated time to completion of the process
import os
import sys
import time
import asyncio

import logging
from hachiko.hachiko import AIOWatchdog, AIOEventHandler
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from InquirerPy import inquirer  # For CLI Input
from pyfiglet import Figlet  # For ASCII art
# for progress bar
from tqdm.asyncio import trange, tqdm

WATCH_DIRECTORY = '/path/to/watch/directory/'
LOG_DIRECTORY = '/path/to/watch/directory/'
progress_bar = None


def main():
    run_app()


# Method for direct launch
def run_app():
    # Print welcome message using PyFiglet
    print_welcome_msg()


def print_welcome_msg():
    # Print welcome message using PyFiglet
    print(Figlet(font='slant').renderText('Progress Monitor'))
    print("Welcome to Progress Monitor, version 0.1.0, by ionite")


# The main menu
def file_actions_menu(path_to_file=None):
    while True:
        # Use the InquirerPy checkboxes, alternate syntax
        process_choices = inquirer.select(
            message="Select the type of monitoring: ",
            choices=["Folder Monitor", "training.log"],
        ).execute()

        # If the user selected the info count action, then run the info count method
        if "Folder Monitor" in process_choices:
            print("Folder Monitor")
            break

        # If the user selected the top used words action, then run the top used words method
        if "training.log" in process_choices:
            print("training.log")
            break


# Method for training.log monitoring
def pre_log_monitor():
    # Check if the file exists in the application directory relative
    # to the current working directory
    while True:
        if os.path.exists(os.path.join(os.getcwd(), "training.log")):
            # If the file exists, then run the file actions menu
            log_monitor()
        else:
            # Using InquirerPy, ask the user for the path to the log file
            path_to_file = inquirer.filepath(
                message="Enter the path to the log file: ",
                default="training.log"
            ).execute()

            # Check if the file exists
            if os.path.exists(path_to_file):
                # If the file exists, then run the file actions menu
                log_monitor(path_to_file)
            else:
                # If the file does not exist, then print an error message and continue
                print("The file does not exist")
                continue


def log_monitor(path_to_file=None):
    # Monitor the log file for any applications writing to it
    # The file is at path_to_file

    asyncio.get_event_loop().run_until_complete(watch_fs(path_to_file))


class MyEventHandler(AIOEventHandler):
    """Subclass of asyncio-compatible event handler."""

    async def on_created(self, event):
        pass
        # print('Created:', event.src_path)  # add your functionality here

    async def on_deleted(self, event):
        pass
        # print('Deleted:', event.src_path)  # add your functionality here

    async def on_moved(self, event):
        pass
        # print('Moved:', event.src_path)  # add your functionality here

    async def on_modified(self, event):
        global progress_bar
        global LOG_DIRECTORY
        # First read the training.log file
        with open(os.path.join(LOG_DIRECTORY, "training.log"), "r") as f:
            # Read the file line by line
            for line in f:
                # If the line contains the word "Extracting baseline pitch data"
                if "Extracting baseline pitch data" in line:
                    # Extract the first integer in the line

                # F


        # If the progress_bar is None, then create a new one
        if progress_bar is None:
            progress_bar = tqdm(total=100)
        # print('Modified:', event.src_path)  # add your functionality here


async def watch_fs(watch_dir):
    evh = MyEventHandler()
    watch = AIOWatchdog(watch_dir, event_handler=evh)
    watch.start()
    for _ in range(20):
        await asyncio.sleep(1)
    watch.stop()


if __name__ == "__main__":
    main()
