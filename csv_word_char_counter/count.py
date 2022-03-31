# This is a simple program that will read from the csv file in the same directory
# The 'text' column will be isolated
# The program will return the word count and character count for the total text

import csv
from tqdm import tqdm
import sys
import mmap
from collections import Counter


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def main():
    # check if a file was supplied by dragging and dropping it onto the program
    if len(sys.argv) < 2:
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
    else:
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r'):
                pass
        except FileNotFoundError:
            print("File not found")
            input("Press enter to exit")
            exit(0)

    # parse the file name to get the file name only
    file_name = file_path.split('/')[-1]

    # detect the delimiator, read the first line, strip quotations, and match first non-alphanumeric character
    delimiter = None
    with open(file_path, 'r') as f:
        line = f.readline().strip('"')
        # build a list of acceptable delimiters
        delimiters_ref = [',', ';', '|', ':']
        for char in line:
            if char in delimiters_ref:
                delimiter = char
                break
        if delimiter is None:
            print("Unable to detect delimiter")
            input("Press enter to exit")
            exit(0)

    # read the header only using csv reader
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        header = next(reader)

    # initialize text_index
    text_index = None

    # get the index of the text column
    if 'text' in header:
        text_index = header.index('text')
    else:
        # ask the user to enter the header name of the text column
        header_name = input(
            "Enter the header name of the text column: ")
        # check if the user entered a valid header name
        if header_name in header:
            text_index = header.index(header_name)
        else:
            print("Invalid header name")
            if input("Do you want to retry? (Y/N): ").upper() == 'Y':
                main()
            else:
                sys.exit(0)

    # Get the total number of lines quickly
    total_lines = get_num_lines(file_path)

    # track counters
    word_count = 0
    char_count = 0
    char_count_no_spaces = 0

    # track the unique words and frequencies
    word_counter = Counter()

    # Main read operations
    print(f"Reading csv: {file_name}:")
    with open(file_path) as file:
        for line in tqdm(file, total=total_lines):
            # seperate line with the delimiter
            text = line.split(delimiter)[text_index]
            # split the text into words
            words = text.split()

            # update counts
            word_count += len(words)
            char_count += len(line)
            char_count_no_spaces += len(line.replace(' ', ''))

            # update unique words
            for word in words:
                if not is_stop_word(word):
                    word_counter[word] += 1

    # print the top 25 unique words, along with their freq
    # freq is printed in both number and percentage of total unique words
    print('\nTop 25 unique words:')
    for word, count in word_counter.most_common(25):
        print(f'{word}: {count:,} ({round(count / word_count * 100, 2)}%)')
    print('-' * 50)

    # print the counter results, with commas for readability
    print(f'Word count: {word_count:,}')
    print(f'Character count: {char_count:,}')
    print(f'Character count not including spaces: {char_count_no_spaces:,}')
    print('-' * 50)

    # estimated processing time in TTS program for all the text lines
    process_names = ['11', '8', '6']
    process_time = [round(total_lines / 11), round(total_lines / 8), round(total_lines / 6)]

    # convert the time to hours, minutes, and seconds and print
    print("Estimated processing time: ")
    for i, dur_sec in enumerate(process_time):
        t_min, t_sec = divmod(dur_sec, 60)
        t_hour, t_min = divmod(t_min, 60)

        # If more than an hour, do not show seconds
        if t_hour > 0:
            print(
                f"    > At {process_names[i]} lines/s : {t_hour:d} hours {t_min:02d} minutes")
        else:
            print(
                f"    > At {process_names[i]} lines/s : {t_min:02d} minutes {t_sec:02d} seconds")

    # Wait for user to press enter
    input('[Press enter to exit]')


# method to check if a word is in the list of stop words
def is_stop_word(word):
    # define the stop words
    stop_words = ["new", "like", "want", "like", "your", "you're", "good", "i", "me", "my", "myself", "we", "our",
                  "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
                  "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
                  "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                  "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
                  "a", "an", "the", "and", "but", "if", "or", "because", "as",
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                  "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                  "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
                  "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                  "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
                  "now"]
    if word.lower() in stop_words:
        return True
    else:
        return False


if __name__ == "__main__":
    main()
