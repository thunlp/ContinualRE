import numpy as np
import sys
import matplotlib.pyplot as plt
import re

path = "./FewRel/"
name_list = ['AGEM',
            'EAEMR',
            'EMR',
            'EWC']



suffix_name = '.txt'
file_list = [name+suffix_name for name in name_list]
markers = ['*-', '*:', '^-', '^:', 'o-', 'o:', 'v-', 'v:', 'x-', 'x:',
           'o--','*--','v--','^--']

num_task = 10

def read_data(file_name):
    file_data = []
    with open(path + file_name) as file_in:
        for line in file_in:
            file_data.append(line.strip())
    avg=file_data[1]
    avg=avg[:-1]
    print(avg)
    avg = avg.split(',')
    avg = [float(item) for item in avg]
    return avg

def print_matrix(matrix):
    for line in matrix:
        for num in list(line):
            if num >=0:
                sys.stdout.write(' %.3f, ' %num)
            else:
                sys.stdout.write('%.3f, ' %num)
        print()

def print_line_avg(matrix):
    line_avg = []
    for line in matrix:
        line_avg.append(sum(line)/len(list(line)))
    sys.stdout.write('avg of each line: ')
    for num in line_avg:
            if num >=0:
                sys.stdout.write(' %.3f, ' %num)
            else:
                sys.stdout.write('%.3f, ' %num)
    print()
    return line_avg

def compute_diff(result, origin):
    diff = []
    for i in range(len(origin)):
        diff.append(result[i] - origin[i])
    return diff

if __name__ == '__main__':
    method_results = []
    method_times = []
    for file_name in file_list:
        avg = read_data(file_name)
        method_results.append(avg)


    for i, avg_result in enumerate(method_results):
        print(avg_result)
        plt.plot(list(range(num_task)), avg_result, markers[i], label =
                 name_list[i])
    plt.legend()
    plt.xlabel('task number', fontsize=18)
    #plt.ylabel('avarage gain on tasks before task i')
    plt.ylabel('avarage accuracy', fontsize=18)
    plt.show()
