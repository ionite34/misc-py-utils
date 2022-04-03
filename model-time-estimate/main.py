# This module will accept an input of a training.log file and parse the information
# to estimate the time it will take to train the model until completion of the
# current stage's target loss delta.
from datetime import datetime
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize, curve_fit
from InquirerPy import inquirer
from pyfiglet import Figlet
from scipy.stats import linregress
from scipy.stats import expon
import iminuit
from probfit import UnbinnedLH
from scipy.optimize import fsolve

# Use plotext to plot data in terminals
import plotext as plt

isDev = True


class ModelTimeEstimate:

    def __init__(self):
        # Current cycles estimate in epochs
        self.epochs_estimate = 0
        # Current estimation for epochs per second
        self.epochs_per_second = 0
        # Current time estimate in seconds
        self.time_estimate = 0
        # Target loss delta
        self.target_loss_delta = 0
        # List of loss delta from graphs.json
        self.loss_delta = []
        # List of checkpoint numbers from log
        self.checkpoint_num = []
        # Matching list of epochs matched to loss delta
        self.epoch_from_log = []
        # List of epochs from graph
        self.epoch_from_graph = []
        # Time code in HH:MM for each epoch stage
        self.time_code_from_log = []
        # Relative time in seconds for each epoch stage, starting from 0
        self.relative_time = []
        # Function
        self.target_power_series = self.func_powerlaw
        # Reverse function
        self.target_reverse_power = self.func_powerlaw_reverse
        # Reverse log function
        self.target_reverse_log = self.func_reverse_log
        # Reverse linear function
        self.target_reverse_linear = self.func_linear_reverse
        # Reverse of preferred function
        self.target_reverse_preferred = None

        # Fit params of primary model
        self.fit = {}
        # Alternate fit params of primary model
        self.fit_log = {}
        # Fit params of secondary time model
        self.fit_time = {}
        # Fit params of preferred fit
        self.fit_preferred = {}

    # Define the log fit function
    @staticmethod
    def func_log(x_in, a, b):
        return np.exp(np.add(np.multiply(a, np.log(x_in)) + b))

    # Define the reverse log fit function
    @staticmethod
    def func_reverse_log(y_in, a, b):
        return np.exp(np.divide(np.multiply(y_in, b), a))

    # Define fit to a custom power law:
    @staticmethod
    def func_powerlaw(x, a, k, b):
        return a * np.exp(-k * x) + b

    # Define the reverse function to the powerlaw to get the x value from the y value
    @staticmethod
    def func_powerlaw_reverse(y_in, a, b):
        pass

    # Define the reverse linear time function
    @staticmethod
    def func_linear_reverse(y_in, m, c):
        return (y_in - c) / m

    # Calculate r squared with np polyfit
    @staticmethod
    def polyfit_r(x_in, y, degree):
        results = {}
        coefficients = np.polyfit(x_in, y, degree)
        p = np.poly1d(coefficients)
        # calculate r-squared
        y_hat = p(x_in)
        y_bar = np.sum(y) / len(y)
        ss_reg = np.sum((y_hat - y_bar) ** 2)
        ss_tot = np.sum((y - y_bar) ** 2)
        results['coefficients'] = [coefficients[1], coefficients[0]]
        results['r_squared'] = ss_reg / ss_tot

        return results

    # Calculate fit using negative log likelihood minimization
    @staticmethod
    def polyfit_new(x_in, y, degree):
        np.random.seed(1)

        def exp_fit(x, N, L):
            return N * np.exp(- L * x)

        def negloglik(args, func, data):
            """Negative log likelihood"""
            return - np.sum(np.log(func(data, *args)))

        def histpoints_w_err(X, bins):
            counts, bin_edges = np.histogram(X, bins=bins, normed=False)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            bin_err = np.sqrt(counts)

            # Generate fitting points
            x = bin_centers[counts > 0]  # filter along counts, to remove any value in the same position as an empty bin
            y = counts[counts > 0]
            sy = bin_err[counts > 0]

            return x, y, sy

        data = np.column_stack((x_in, y))
        bins = np.arange(0, 3, 0.1)

        x, y, sy = histpoints_w_err(data, bins)
        popt, pcov = curve_fit(exp_fit, x, y, sigma=sy)

        xpts = np.linspace(0, 3, 100)

        # All variables must be positive
        bnds = ((0, None),
                (0, None))

        # minimize the negative log-Likelihood
        result = minimize(negloglik, args=(exp_fit, data), x0=np.array(15, 0.5))

        jac = result.get("jac")

    # Calculate preferred fit and function
    def calculate_preferred_fit(self):
        print('Calculating preferred fit...')

        # Compare between power and log for r_squared
        diff = None
        if not np.isnan(self.fit_log['r_squared']) and not np.isnan(self.fit['r_squared']):
            diff = np.subtract(self.fit_log['r_squared'], self.fit['r_squared'])

        if diff >= 0:
            self.fit_preferred = self.fit_log
            self.target_reverse_preferred = self.target_reverse_log
            print(f'Delta R-squared: {np.around(np.abs(diff), 5)}')
            print(f'Preferred fit: Logarithmic')
        elif diff < 0:
            self.fit_preferred = self.fit
            self.target_reverse_preferred = self.target_reverse_power
            print(f'Delta R-squared: {np.around(np.abs(diff), 5)}')
            print(f'Preferred fit: Power Series')
        elif diff is None and np.isnan(self.fit['r_squared']) and not np.isnan(self.fit_log['r_squared']):
            self.fit_preferred = self.fit_log
            self.target_reverse_preferred = self.target_reverse_log
            print(f'Power series fit invalid.')
            print(f'Preferred fit: Logarithmic')
        elif diff is None and np.isnan(self.fit_log['r_squared']) and not np.isnan(self.fit['r_squared']):
            self.fit_preferred = self.fit
            self.target_reverse_preferred = self.target_reverse_power
            print(f'Logarithmic fit invalid.')
            print(f'Preferred fit: Power Series')
        else:
            print('Both fits are invalid. No preferred fit found')
            input('Press enter to exit...')
            exit(0)

    # Calculates the relative time list from the time_code_from_log list
    def calculate_relative_time(self):
        # Iterate through the time code list
        for i in range(len(self.time_code_from_log)):
            # If we are at the first element
            if i == 0:
                # Set the relative time to 0
                self.relative_time.append(0)
            else:
                # Get the time code for the first element
                first_time_code = self.time_code_from_log[0]
                time_code = self.time_code_from_log[i]
                # Strip time into datetime objects
                first_time = datetime.strptime(first_time_code, '%H:%M')
                time = datetime.strptime(time_code, '%H:%M')
                # Calculate the difference in time
                time_diff = time - first_time
                # Convert to seconds
                time_diff_seconds = time_diff.total_seconds()
                # Add to the relative time list
                self.relative_time.append(time_diff_seconds)

    @staticmethod
    def unbinned_exp_LLH(data, loc_init, scale_init, limit_loc, limit_scale):
        # Define function to fit
        def exp_func(x, loc, scale):
            return expon.pdf(x, loc, scale)

        # Define initial parameters
        init_params = dict(loc=loc_init, scale=scale_init)

        # Create an unbinned likelihood object with function and data.
        unbin = probfit.UnbinnedLH(exp_func, data)

        # Minimizes the unbinned likelihood for the given function
        m = iminuit.Minuit(unbin,
                           **init_params,
                           limit_scale=limit_loc,
                           limit_loc=limit_scale,
                           pedantic=False,
                           print_level=0)
        m.migrad()
        params = m.values.values()  # Get out fit values
        errs = m.errors.values()
        return params, errs

    # Calculates the logarithmic fit to the loss_delta vs relative_time for loss delta over time
    def calculate_fit(self):
        # Currently the size of loss_delta is much larger than relative_time
        # Because the graph has more info points than the log file
        # We will use the self.relative_time list and matching self.checkpoint_num list
        # And find the match between checkpoint_num (from log) and epoch_from_graph (from graph)

        # We will build a new list of epoch_uniform which is the same size as self.relative_time
        loss_delta_uniform = []

        for index, time_at_i in enumerate(self.relative_time):
            # Get the matching checkpoint_num from self.checkpoint_num
            checkpoint_from_log = self.checkpoint_num[index]
            if checkpoint_from_log in self.epoch_from_graph:
                # Search for the matching checkpoint_num in self.epoch_from_graph
                index_of_match = self.epoch_from_graph.index(checkpoint_from_log)
                # Use this index within the loss_delta list to get the loss_delta
                current_loss_delta = self.loss_delta[index_of_match]
                # Add the matching epoch to the epoch_uniform list
                loss_delta_uniform.append(current_loss_delta)
            else:
                print(f'[WARN] No matching epoch found for checkpoint {checkpoint_from_log}')

        # Plot the (y) loss_delta vs (x) epoch_from_graph
        # We will plot a shifted power series relationship to determine
        # the estimated epochs until the loss delta is less than the target loss delta
        loss_delta_array = np.array(self.loss_delta)
        epoch_from_graph_array = np.array(self.epoch_from_graph)
        # We will first shift the epoch_from_graph_array by the first element, such it starts at 0
        epoch_from_graph_array_shifted = epoch_from_graph_array - epoch_from_graph_array[0]

        dataset_1 = np.column_stack((epoch_from_graph_array_shifted, loss_delta_array))

        # We will also need to plot the (y) checkpoint_num vs (x) relative_time
        # This should be a linear relationship
        # We will use the resultant function to determine the epochs per second (gradient)
        checkpoint_num_from_log_array = np.array(self.checkpoint_num)
        relative_time_array = np.array(self.relative_time)
        # Replace first element of relative_time_array with 0.01
        relative_time_array[0] = 0.01
        # Shift the checkpoint_num_from_log_array by the first element, such it starts at 1
        checkpoint_num_from_log_array_shifted = checkpoint_num_from_log_array - checkpoint_num_from_log_array[0] + 1
        dataset_2 = np.column_stack((relative_time_array, checkpoint_num_from_log_array_shifted))

        # Drop any NaN value rows
        dataset_1 = dataset_1[~np.isnan(dataset_1).any(axis=1)]
        dataset_2 = dataset_2[~np.isnan(dataset_2).any(axis=1)]

        # Use only the first 20 rows of the dataset
        # dataset_1 = dataset_1[:20]

        # Force set first value to 1
        dataset_1[0, 0] = 1

        # Print Data
        # print('/n')
        # np.set_printoptions(suppress=True)
        # print(f'[INFO] dataset_1: {dataset_1}')
        # print('/n')
        # print(f'[INFO] dataset_2: {dataset_2}')
        # print('/n')

        # Calculate fit for the primary dataset
        x = dataset_1[:, 0]
        y = dataset_1[:, 1]

        # Plot
        plt.plot(x, y)
        plt.xscale('log')
        plt.yscale('linear')
        plt.grid(1, 0)
        plt.title('Loss Delta over Iterations')
        plt.xlabel('Iterations (Log Scale)')
        plt.ylabel('Loss Delta')
        plt.show()

        # REVERSE POWER SERIES FIT
        p0 = (1., 1.e-5, 1.)
        est_fit = curve_fit(self.target_power_series, x, y, p0, maxfev=10000)
        # Extract the fit parameters
        popt, pcov = est_fit
        self.fit['coefficients'] = popt
        # Get the residual sum of squares
        residuals = y - self.target_power_series(x, *popt)
        ss_res = np.sum(residuals ** 2)
        # Get the total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # Get the r-squared
        r_squared = 1 - (ss_res / ss_tot)
        self.fit['r_squared'] = r_squared
        print(f'[INFO] power r_squared: {r_squared}')

        # LOG FIT
        log_fit = self.polyfit_r(np.log(x), y, 1)
        self.fit_log = log_fit
        print(f"[INFO] Log fit coefficients: {log_fit['coefficients']}")
        print(f"[INFO] Log fit r-squared: {log_fit['r_squared']}")

        # print(f'(loss delta) = {popt[2]} + {popt[1]} * (epochs) ^ {popt[0]}')
        print(f'The target loss delta is {self.target_loss_delta}')
        print('\n')

        # Plot
        # plt.scatter(x, y, label='From Data')
        # plt.plot(x, self.target_reverse_log(x, *log_fit['coefficients']), label='Inverse Power Series Fit')
        # plt.grid(1, 0)
        # plt.title('Loss Delta over Iterations')
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss Delta')
        # plt.show()

        # Calculate fit for the secondary dataset, which is the relative time against checkpoint_num
        x2 = dataset_2[:, 0]
        y2 = dataset_2[:, 1]

        fit_time_raw = linregress(x2, y2)
        slope, intercept, r, p, se = fit_time_raw
        self.fit_time['coefficients'] = [slope, intercept]
        self.fit_time['r_squared'] = r ** 2

        # Get the r-squared
        r_squared2 = r ** 2
        self.epochs_per_second = slope
        print(f'Secondary linear time series fit r_squared: {r_squared2}')
        print(f'(epochs) = {slope} * (relative time) + {intercept}')
        print('\n')

    # Calculate the epochs until the loss delta is less than the target loss delta
    def calculate_epochs_until_loss_delta(self):
        # Use the reverse model input to calculate the epochs
        func = self.target_reverse_preferred
        t_loss_delta = self.target_loss_delta
        estimated_epochs = func(t_loss_delta, *self.fit_preferred['coefficients'])
        self.epochs_estimate = estimated_epochs

    # Calculates the time in seconds estimate based on epochs_until_loss_delta and epochs_per_second
    def calculate_time_estimate(self):
        main_func = self.target_reverse_preferred
        sec_func = self.target_reverse_linear
        iter_est = main_func(self.target_loss_delta, *self.fit_preferred['coefficients'])
        time_est = sec_func(iter_est, *self.fit_time['coefficients'])
        print(f'Estimated time: {time_est} seconds')
        print(f'Estimated iterations: {iter_est}')
        # time_est = self.epochs_estimate / self.epochs_per_second
        self.time_estimate = int(round(time_est))
        return self.get_formatted_time_estimate(self.time_estimate)

    # Using the last relative time, calculate the time remaining
    def calculate_time_remaining(self):
        # Get the last relative time
        last_relative_time = self.relative_time[-1]
        # Check if time_remaining is more than the current last relative time
        if self.time_estimate > last_relative_time:
            # Use total time_estimate - last_relative_time to get the time remaining
            time_remaining = self.time_estimate - last_relative_time
            # Get formatted time remaining
            return self.get_formatted_time_estimate(time_remaining)
        else:
            return 'Less than current time'

    # Method to get formatted time estimate string from seconds
    @staticmethod
    def get_formatted_time_estimate(seconds_in):
        # Convert to DD:HH:MM
        # Get the number of days
        days = int(seconds_in / 86400)
        # Using remainder, get the number of hours
        hours = int((seconds_in % 86400) / 3600)
        # Using remainder, get the number of minutes
        minutes = int(((seconds_in % 86400) % 3600) / 60)

        # Convert to string
        days_str = str(days)
        hours_str = str(hours)
        minutes_str = str(minutes)

        if days > 0:
            return days_str + " days, " + hours_str + " hours, " + minutes_str + " minutes"
        elif hours > 0:
            return hours_str + " hours, " + minutes_str + " minutes"
        elif minutes > 0:
            return minutes_str + " minutes"
        else:
            return "less than a minute"


def main():
    # First check if the user is directly running the app or from dragging a file onto the executable
    if len(sys.argv) > 1:
        # If the user is running via drag
        run_via_drag()
    else:
        # If direct launch
        run_app()


# Method to run the quit prompt
def run_quit_prompt():
    choice = inquirer.select(message="Press enter to exit:", choices=["Exit"]).execute()
    if choice == "Exit":
        sys.exit(0)


# Method for direct launch
def run_via_drag():
    # Print welcome message using PyFiglet
    print_welcome_msg()

    # grab the file path from sys args
    file_path = sys.argv[1]

    # check if we can read the file
    if not os.path.isfile(file_path):
        print("Error: File does not exist")
        input("Press enter to exit...")
        sys.exit(1)

    # run the main method
    main_operations(file_path)


# Method for direct launch
def run_app():
    global isDev
    # Print welcome message using PyFiglet
    print_welcome_msg()

    # print message
    print("> You can also drag the training.log file onto this executable to start it automatically.")
    while True:
        if isDev:
            file_path = "L:/Voice ML Workspace/Current Trainer Outputs/test_timings/training.log"
            break
        # Ask the user for the file path
        file_path = inquirer.filepath(message="Enter the file path of the training.log file: ").execute()
        # If any quotation marks are present, remove them
        file_path = file_path.replace('"', '')
        # Check if the file exists
        if not os.path.isfile(file_path):
            print("Error: File does not exist")
            # Using InquirerPy, choose retry or exit
            choice = inquirer.select(message="Choose an option:", choices=["Retry", "Exit"]).execute()
            if choice == "Retry":
                continue
            else:
                sys.exit(0)
        else:
            break

    # run the main method
    main_operations(file_path)


def main_operations(file_path):
    # Creates a new instance of ModelTimeEstimate
    mte = ModelTimeEstimate()

    # lines list
    lines = []

    # Read the log file until the end of file, store the lines in a list
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Using the last line, find the index of "Stage: "
    stage_index = lines[-1].find("Stage: ")
    end_of_stage = len(lines) - 2
    start_of_stage = None

    # Using last line, find target loss delta, index of "Target: "
    target_loss_delta_index = lines[-1].find("Target: ")
    mte.target_loss_delta = lines[-1][target_loss_delta_index + 8:]

    # If not found, print error message and exit
    if stage_index == -1:
        print("Error: Could not find stage in log file")
        input("Press enter to exit...")
        sys.exit(1)

    # Get the stage number from the last line and store
    stage_num = lines[-1][stage_index + 7]

    # Starting from the last line, iterate through lines until index 0
    # Until we find a line that does not contain "Stage: "
    for i in range(end_of_stage, 0, -1):
        # If the line does not contain "Stage: "
        if lines[i].find("Stage: ") == -1:
            # Safety check to make sure we're not on the first line
            if i != 0:
                # Get the line index of the previous line
                start_of_stage = i + 1
                # Add another line to start at the 2nd line for some weird graph issue
                start_of_stage += 1
                # Break out of the loop
                break

    # If start_of_stage is None, print error message and exit
    if start_of_stage is None:
        print("Error: Could not find start of stage in log file")
        inquirer.confirm(message="Press any key to exit").execute()
        sys.exit(1)

    # Iterate through lines from the start of the stage to the end of the stage
    # Flag for invalid file format
    invalid_file = False
    for i in range(start_of_stage, end_of_stage):
        # Split the line using the delimiter " | "
        line_split = lines[i].split(" | ")
        # Parse Time Code
        # Record the first element of the split line
        time_code = line_split[0]
        # Add to object
        mte.time_code_from_log.append(time_code)

        # Parse Epoch
        # Record the integer after "Epoch: "
        epoch_num = line_split[1][line_split[1].find("Epoch: ") + 7] if line_split[1].find("Epoch: ") != -1 else None

        # Parse the 4th element, the file checkpoint name
        file_checkpoint_name = line_split[3]

        # Iterate per character from the end of the string to the beginning
        # We are looking to record the index of the first "." and the next "_"
        symbol1_index, symbol2_index = None, None
        for index, char_i in enumerate(reversed(file_checkpoint_name)):
            # If we find a "."
            if char_i == ".":
                # Record the index of the "."
                symbol1_index = len(file_checkpoint_name) - index - 1
                symbol1_index = int(symbol1_index)
            # If we find a "_"
            elif char_i == "_":
                # Record the index of the "_"
                symbol2_index = len(file_checkpoint_name) - index - 1
                symbol2_index = int(symbol2_index)
                break

        # If we found both "." and "_"
        if symbol1_index is not None and symbol2_index is not None:
            # Grab the section of string between these indexes
            file_checkpoint_num = file_checkpoint_name[(symbol2_index + 1):symbol1_index]
            # Fit to integer
            file_checkpoint_num = int(file_checkpoint_num)
            # Add to object
            mte.checkpoint_num.append(file_checkpoint_num)
        # If we did not find both "." and "_"
        else:
            # Print Error Message and exit
            print("Error: Could not find file checkpoint number in log file")
            inquirer.confirm("Press any key to exit")
            sys.exit(1)

    # Now find the Dataset directory using function
    dataset_dir = find_dataset_dir(lines)

    # We will try to read the graphs.json file in the same folder as the log file
    # If the file does not exist, report error and exit
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    graphs_json_path = os.path.join(parent_dir, "graphs.json")
    if not os.path.isfile(graphs_json_path):
        print(f"Error: Could not find graphs.json file in dataset directory {graphs_json_path}")
        inquirer.confirm(message="Press any key to exit").execute()
        sys.exit(1)
    else:
        pass

    # Read the graphs.json file
    with open(graphs_json_path, "r") as f:
        graphs_json = json.load(f)
        # Look for "stages" / <stage_num> / "loss_delta" and grab the list of lists
        loss_delta_list = graphs_json["stages"][stage_num]["loss_delta"]
        # Look for "stages" / <stage_num> / "target_delta" and grab the number
        target_delta = graphs_json["stages"][stage_num]["target_delta"]

        # Set the target_delta in the ModelTimeEstimate object
        mte.target_loss_delta = target_delta
        # Parse the list of loss_delta_list. The format is [epoch, loss_delta]
        for i in range(len(loss_delta_list)):
            # Grab the epoch number
            epoch_num = loss_delta_list[i][0]
            # Grab the loss delta
            loss_delta = loss_delta_list[i][1]
            # Set to object
            mte.epoch_from_graph.append(epoch_num)
            mte.loss_delta.append(loss_delta)

    # Call calculations
    mte.calculate_relative_time()
    mte.calculate_fit()
    mte.calculate_preferred_fit()
    mte.calculate_epochs_until_loss_delta()
    mte.calculate_time_estimate()
    time_total_string = mte.calculate_time_estimate()
    time_remaining_string = mte.calculate_time_remaining()

    # Print the results
    print("\n")
    print("Stage: " + str(stage_num))
    print("Time Estimate Total: " + time_total_string)
    print("Time Estimate Remaining: " + time_remaining_string)
    print("\n")
    # press any key to exit
    run_quit_prompt()


def find_dataset_dir(lines):
    # Iterate through lines
    for line in lines:
        # If the line contains '| Dataset: '
        if line.find("| Dataset: ") != -1:
            # Get the string after "| Dataset: "
            # Remove any trailing new lines
            dataset_dir = line[line.find("| Dataset: ") + 11:]
            dataset_dir = dataset_dir.rstrip()
            # Return the string
            return dataset_dir

    # If we get to this point, then the file is invalid
    return None


def print_welcome_msg():
    # Print welcome message using PyFiglet
    print(Figlet(font='slant').renderText('Training Time Estimator'))
    print("Welcome to Training Time Estimator, v0.1.1")
    print("by https://github.com/ionite34/")


if __name__ == '__main__':
    main()
