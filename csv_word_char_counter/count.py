# This is a simple program that will read from the csv file in the same directory
# The 'text' column will be isolated
# The program will return the word count and character count for the total text

import csv
import sys
    
def main():
    # Automatically grabs the file directory of the file dragged into the program

    # If there is no file name, the program will exit
    if len(sys.argv) < 2:
        print("Please drag a file into the program")
        input('[Press enter to exit]')
        sys.exit(0)
    else:
        file_dir = sys.argv[1]
        # parse the file name to get the file name only

    file_name = file_dir.split('/')[-1]

    # open the file
    with open(file_dir, 'r') as csvfile:
        # create a csv reader object
        csvreader = csv.reader(csvfile)
        # read the header row
        header = next(csvreader)
        
        # find the column called 'text' and the index by reading header
        # first check if the 'text' column exists
        if 'text' in header:
            text_index = header.index('text')
        else:
            # ask the user to enter the header name of the text column
            header_name = input("Enter the header name of the text column: ")
            # check if the user entered a valid header name
            if header_name in header:
                text_index = header.index(header_name)
            else:
                print("Invalid header name")
                # Y to continue, N to exit
                if input("Do you want to retry? (Y/N): ").upper() == 'Y':
                    main()
                else:
                    sys.exit(0)

        # calculate the total number of lines
        total_lines = sum(1 for line in open(file_dir))
        # track word count
        word_count = 0
        # track character count
        char_count = 0
        # track character count not including spaces
        char_count_no_spaces = 0
        
        # read each row of data
        for row in csvreader:
            #spinner.next()
            # isolate the text column
            text = row[text_index]
            # split the text into words
            words = text.split()
            # update the word count
            word_count += len(words)
            # update the character count
            char_count += len(text)
            # update the character count not including spaces
            char_count_no_spaces += len(text.replace(' ', ''))

        # print the results, with commas for readability
        print(f'Word count: {word_count:,}')
        print(f'Character count: {char_count:,}')
        print(f'Character count not including spaces: {char_count_no_spaces:,}')

        # calculate the estimated processing time in TTS program for all the text lines
        # store in process_time list
        process_names = ['11', '8', '6']
        process_time = []
        process_time.append(round(total_lines / 11))
        process_time.append(round(total_lines / 8))
        process_time.append(round(total_lines / 6))

        # convert the time to hours, minutes, and seconds and print
        print("Estimated processing time: ")
        for i, dur_sec in enumerate(process_time):
            min, sec = divmod(dur_sec, 60)
            hour, min = divmod(min, 60)
            
            # If more than an hour, do not show seconds
            if hour > 0:
                print(f"    > At {process_names[i]} lines/s : {hour:d} hours {min:02d} minutes")
            else:
                print(f"    > At {process_names[i]} lines/s : {min:02d} minutes {sec:02d} seconds")

        # Wait for user to press enter
        input('[Press enter to exit]')


if __name__ == "__main__":
    main()
