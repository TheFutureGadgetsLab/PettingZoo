import matplotlib.pyplot as plt
from sys import argv
from getopt import getopt, GetoptError
import pandas as pd

def main():
    try:
        opts, run_dir = getopt(argv[1:], "hmadct", ["help", 'max', 'average', 'died', 'completed', 'timedout'])
    except GetoptError as err:
        print(err)
        usage()
        exit(2)
    
    headers = ['min', 'max', 'average', 'died', 'completed', 'timedout']
    used_cols = []
    for opt, _ in opts:
        if opt in ("-h", "--help"):
            usage()
            return
        elif opt in ("-m", "--max"):
            used_cols.append(1)
        elif opt in ("-a", "--average"):
            used_cols.append(2)
        elif opt in ("-d", "--died"):
            used_cols.append(3)
        elif opt in ("-c", "--completed"):
            used_cols.append(4)
        elif opt in ("-t", "--timedout"):
            used_cols.append(5)
        else:
            print("Unknown option!")
            usage()
            return
    
    if len(run_dir) != 1:
        print("Please provide a run directory!")
        usage()
        exit(2)

    while 1:
        plt.clf()
        ax = plt.gca()
        stats = pd.read_csv(f'{run_dir[0]}/run_log.txt', names=headers, usecols=used_cols)
        stats.plot(ax=ax)
        plt.pause(1)

def usage():
    print('Usage: python3 visualise_runs.py [-a] [-d] [-t] [-c] rundir')
    print('  -a, --average      Display plot of average fitness per generation')
    print('  -m, --max          Display plot of max fitness per generation')
    print('  -d, --died         Display plot of death count per generation')
    print('  -t, --timedout     Display plot of timedout runs per generation')
    print('  -c, --completed    Display plot of completed runs per generation')

if __name__ == "__main__":
    main()