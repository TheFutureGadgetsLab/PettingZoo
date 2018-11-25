from numpy import genfromtxt
import matplotlib.pyplot as plt
from sys import argv, exit
from getopt import getopt, GetoptError

def main():
    try:
        opts, args = getopt(argv[1:], "hatcd", ["help", "average", "timedout", "completed", "died"])
    except GetoptError as err:
        print(err)
        usage()
        exit(2)

    if len(args) == 0:
        print("Please supply at least one run directory!")
        usage()
        exit(2)
    
    average = False
    timedout = False
    completed = False
    died = False
    for opt, _ in opts:
        if opt in ("-h", "--help"):
            usage()
            exit()
        elif opt in ("-a", "--average"):
            average = True
        elif opt in ("-t", "--timedout"):
            timedout = True
        elif opt in ("-c", "--completed"):
            completed = True
        elif opt in ("-d", "--died"):
            died = True 
        else:
            assert False, "unhandled option"

    runs, run_headers = load_data(args)

    if average:
        plot_avg_fit(runs, run_headers, 1)
    if timedout:
        plot_timeout(runs, run_headers, 2)
    if completed:
        plot_completed(runs, run_headers, 3)
    if died:
        plot_died(runs, run_headers, 4)

    plt.show()

def load_data(run_files):
    run_headers = []
    runs = []

    try:
        for run in run_files:
            # Populate header list
            # Cols: IN_H, IN_W, HLC, NPL, GEN_SIZE, GENERATIONS, MUTATE_RATE
            run_headers.append(genfromtxt(f'./{run}/run_data.txt', delimiter=',', max_rows = 1, dtype=int))
            # Populate data list
            # Cols: completed, timed out, died, average, max, min
            runs.append(genfromtxt(f'./{run}/run_data.txt', delimiter=',', skip_header=1, dtype=None))
    except IOError as err:
        print(err)
        exit(1)

    return runs, run_headers

def plot_completed(runs, run_headers, n):
    plt.figure(n)

    plt.xlabel('Generations')
    plt.ylabel('Completed')
    plt.title('Agents Completing over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[0] for point in run], label=run_label)

    plt.legend(loc='best')

    plt.xlim(left=0)
    plt.grid(True)

def plot_timeout(runs, run_headers, n):
    plt.figure(n)

    plt.xlabel('Generations')
    plt.ylabel('Timed Out')
    plt.title('Agents Timing Out over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[1] for point in run], label=run_label)

    plt.legend(loc='best')

    plt.xlim(left=0)
    plt.grid(True)

def plot_died(runs, run_headers, n):
    plt.figure(n)

    plt.xlabel('Generations')
    plt.ylabel('Died')
    plt.title('Agents dying over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[2] for point in run], label=run_label)

    plt.legend(loc='best')

    plt.xlim(left=0)
    plt.grid(True)

def plot_avg_fit(runs, run_headers, n):
    plt.figure(n)

    plt.xlabel('Generations')
    plt.ylabel('Average fitness')
    plt.title('Avg. Fitness over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[3] for point in run], label=run_label)

    plt.legend(loc='best')

    plt.xlim(left=0)
    plt.grid(True)

def update_avg_fit(runs, run_headers, n):
    plt.figure(n)

    # Grab each line in the current axis and update its y data
    for run, line in zip(runs, plt.gca().get_lines()):
        line.set_ydata([point[3] for point in run])

def usage():
    print(f'Usage: python3 {argv[0]}')

if __name__ == "__main__":
    main()
