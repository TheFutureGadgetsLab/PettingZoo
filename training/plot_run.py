from numpy import genfromtxt
import matplotlib.pyplot as plt
from sys import argv, exit
from getopt import getopt, GetoptError
from tkinter import TclError
import matplotlib

def main():
    try:
        opts, args = getopt(argv[1:], "hatcdm", ["help", "average", "timedout", "completed", "died", "max"])
    except GetoptError as err:
        print(err)
        usage()
        exit(2)
    
    indices = {'min': 0, 'max': 1, 'average': 2, 'died': 3, 'completed': 4, 'timedout': 5}

    average = False
    timedout = False
    completed = False
    died = False
    max_ = False
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
        elif opt in ("-m", "--max"):
            max_ = True
        elif opt in ("-d", "--died"):
            died = True 
        else:
            assert False, "unhandled option"
    
    if len(args) == 0:
        print("Please supply at least one run directory!")
        usage()
        exit(2)

    runs, run_headers = load_data(args)

    plt.ion()
    plt.show()
            
    if average:
        plot_avg_fit(runs, run_headers, indices['average'])
    if timedout:
        plot_timeout(runs, run_headers, indices['timedout'])
    if completed:
        plot_completed(runs, run_headers, indices['completed'])
    if died:
        plot_died(runs, run_headers, indices['died'])
    if max_:
        plot_max(runs, run_headers, indices['max'])

    while len(plt.get_fignums()) > 0:
        # Data is re-read every loop (1 second), probably unnecessary but not super expensive
        runs, run_headers = load_data(args)

        if average:
            update_plot(runs, run_headers, indices['average'])
        if timedout:
            update_plot(runs, run_headers, indices['timedout'])
        if completed:
            update_plot(runs, run_headers, indices['completed'])
        if died:
            update_plot(runs, run_headers, indices['died'])
        if max_:
            update_plot(runs, run_headers, indices['max'])

        # Try block to handle user closing figure during pause
        try:
            # This will pause (not blocking GUI loop) and update plots
            mypause(1)
        except TclError:
            pass

def load_data(run_files):
    run_headers = []
    runs = []

    try:
        for run in run_files:
            # Populate header list
            run_headers.append(genfromtxt(f'{run}/run_log.txt', delimiter=',', max_rows = 1, dtype=int))
            # Populate data list
            runs.append(genfromtxt(f'{run}/run_log.txt', delimiter=',', skip_header=1, dtype=None))
    except IOError as err:
        print(err)
        exit(1)

    return runs, run_headers

def plot_completed(runs, run_headers, index):
    plt.figure(index)

    plt.xlabel('Generations')
    plt.ylabel('Completed')
    plt.title('Agents Completing over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[0] for point in run], label=run_label)

    plt.legend(loc='best')
    plt.grid(True)

def plot_timeout(runs, run_headers, index):
    plt.figure(index)

    plt.xlabel('Generations')
    plt.ylabel('Timed Out')
    plt.title('Agents Timing Out over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[1] for point in run], label=run_label)

    plt.legend(loc='best')
    plt.grid(True)

def plot_died(runs, run_headers, index):
    plt.figure(index)

    plt.xlabel('Generations')
    plt.ylabel('Died')
    plt.title('Agents dying over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[2] for point in run], label=run_label)

    plt.legend(loc='best')
    plt.grid(True)

def plot_avg_fit(runs, run_headers, index):
    plt.figure(index)

    plt.xlabel('Generations')
    plt.ylabel('Average fitness')
    plt.title('Avg. Fitness over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[3] for point in run], label=run_label)

    plt.legend(loc='best')
    plt.grid(True)

def plot_max(runs, run_headers, index):
    plt.figure(index)

    plt.xlabel('Generations')
    plt.ylabel('Max fitness')
    plt.title('Max Fitness over Time')

    for run, hdr in zip(runs, run_headers):
        run_label = f'Input: {hdr[0]}x{hdr[1]}\nHLC: {hdr[2]}\nNPL: {hdr[3]}\nGen size: {hdr[4]}\nGen count: {hdr[5]}'
        plt.plot([point[1] for point in run], label=run_label)

    plt.legend(loc='best')
    plt.grid(True)

def update_plot(runs, run_headers, index):
    plt.figure(index)

    # Grab each line in the current axis and update its y data
    for run, line in zip(runs, plt.gca().get_lines()):
        line.set_ydata([point[index] for point in run])
        line.set_xdata(range(len(run)))
    
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view()
    plt.autoscale(enable=True)

def usage():
    print('Usage: python3 visualise_runs.py [-a] [-d] [-t] [-c] rundir1 [rundir2]...')
    print('  -a, --average      Display plot of average fitness per generation')
    print('  -d, --died         Display plot of death count per generation')
    print('  -t, --timedout     Display plot of timedout runs per generation')
    print('  -c, --completed    Display plot of completed runs per generation')
    print('  -m, --max    Display plot of completed runs per generation')

def mypause(interval):
    "Similar to plt.pause(), however, it does not bring fig into foreground."
    
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

if __name__ == "__main__":
    main()
