import csv
import sys
import time
import numpy as np


def open_csv(filename, delimiter=';', encoding="utf8"):
    result = []
    with open(filename, mode="rt", encoding=encoding) as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        for row in reader:
            if row:
                result.append(row)
    return result


def split_dataset(x_dataset: np.array, y_dataset: np.array, train_ratio: float):
    arr = np.arange(x_dataset.size)
    np.random.shuffle(arr)
    num_train = int(train_ratio * x_dataset.size)
    x_train = x_dataset[arr[:num_train]]
    x_test = x_dataset[arr[num_train:]]
    y_train = y_dataset[arr[:num_train]]
    y_test = y_dataset[arr[num_train:]]
    return x_train, x_test, y_train, y_test


def printProgressBar(iteration, total, prefix='Progress:', suffix='', decimals=1, length=100, fill='█', lost='-', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + lost * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


# x_dataset = np.linspace(-1, 1, 101)
# y_dataset = 2 * x_dataset + np.random.randn(*x_dataset.shape) * 0.33 + 1
#
# x_train, x_test, y_train, y_test = split_dataset(x_dataset, y_dataset, 0.7)
# print(x_train, x_test, y_train, y_test, sep='\n')

def say(word):
    print('\n'.join([''.join([(word[(x-y) % len(word)] if ((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3 <= 0 else ' ') for x in range(-32, 32)]) for y in range(17, -17, -1)]))
# say('HiPython') say('_Python')


if __name__ == "__main__":
    # tensorboard --logdir=../logs
    items = list(range(0, 137))
    printProgressBar(0, len(items), suffix='Starting', length=65)
    for i, item in enumerate(items):
        time.sleep(0.06)
        printProgressBar(i, len(items), suffix='Loading...', length=65)
    printProgressBar(1, 1, suffix='Complete!', length=65)
    # print(*open_csv("datasets/iris/iris.data.csv", delimiter=','), sep='\n')
