# Main entry point for the CLI

# Uses the PyInquirer library to create a CLI
from __future__ import print_function, unicode_literals
from InquirerPy import inquirer  # For CLI input
from pyfiglet import Figlet  # For ASCII art

# Import the RegisterContext class
from csv_word_char_counter.register_context import RegisterContext


def run_context_menu(__file__):
    print_welcome_msg()
    file_actions_menu(__file__)


def print_welcome_msg():
    # Print welcome message using PyFiglet
    print(Figlet(font='slant').renderText('CSV Analyzer'))
    print("Welcome to CSV Analyzer, version 1.0.0")


def main():
    # First check if the user is directly running the app or via context menu
    if __name__ == '__main__':
        # If the user is running the app directly, then run the main method
        run_app()
    else:
        # If the user is running the app via context menu, then run the context menu method
        # Get the file path the context menu was called on and pass to the method
        run_context_menu(__file__)


# Method for direct launch
def run_app():
    # Print welcome message using PyFiglet
    print_welcome_msg()

    # We will check if the program is correctly registered in the
    # Windows registry as a context option. If not, ask the user if they want to register it.
    reg_options()


# Method for entering file name manually
def query_file_name_manual():
    # check if a file was supplied by dragging and dropping it onto the program
    print("Note you can also drag a file onto the program to run it")
    file_path = input('Enter path to file: ')
    # trim any quotations from the input, if they exist
    file_path = file_path.strip('"')
    # check that the file exists
    try:
        with open(file_path, 'r'):
            pass
    except FileNotFoundError:
        print("File not found")
        input("Press enter to exit")
        exit(0)

    try:
        with open(file_path, 'r'):
            pass
    except FileNotFoundError:
        print("File not found")
        input("Press enter to exit")
        exit(0)

    # parse the file name to get the file name only
    file_name = file_path.split('/')[-1]
    # return the file name
    return file_name, file_path


def reg_options():
    # Create register_context object
    register_context = RegisterContext()
    # Check if the program is registered
    if not register_context.is_registered:
        # If the program is not registered, ask the user if they want to register it
        # Using InquirerPy
        print("> CSV Analyzer is not registered as a Windows context menu option.")
        confirm = inquirer.confirm(message="Register now?").execute()
        if confirm:
            # If the user confirms, register the program
            register_context.register()
            print("> CSV Analyzer has been registered as a Windows context menu option.")
            print("Press any key to exit...")
            input()
            exit(0)
        else:
            # If the user does not confirm, exit the program
            print("> ...")
            exit(0)
    else:
        # If the program is registered, ask the user if they want to unregister it
        # Using InquirerPy
        print("> CSV Analyzer is registered as a Windows context menu option.")
        confirm = inquirer.confirm(message="Do you wish to unregister?").execute()
        if confirm:
            # If the user confirms, unregister the program
            register_context.unregister()
            print("> CSV Analyzer has been unregistered as a Windows context menu option.")
            print("Press any key to exit...")
            input()
            exit(0)
        else:
            # If the user does not confirm, exit the program
            print("> Exiting...")
            exit(0)


# The main menu for CSV operations
def file_actions_menu(path_to_file=None):
    while True:
        # Use the InquirerPy checkboxes, alternate syntax
        process_choices = inquirer.checkbox(
            message="Select the file actions you want to perform:",
            choices=["Info Count", "Top used words", "Estimate Processing time"],
        ).execute()

        # Check that at least one action was selected
        if len(process_choices) == 0:
            print("You must select at least one action.")
            continue

        # If the user selected the info count action, then run the info count method
        if "Info Count" in process_choices:
            print("Info Count")

        # If the user selected the top used words action, then run the top used words method
        if "Top used words" in process_choices:
            print("Top used words")

        # If the user selected the estimate processing time action, then run the estimate processing time method
        if "Estimate Processing time" in process_choices:
            print("Estimate Processing time")


if __name__ == "__main__":
    main()
