# python3.6

import csv


def read_csv(filename, headers, filter_item_list=[]):
    data_list = []
    with open(filename, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = dict(row)
            for item in filter_item_list:
                data.pop(item, None)
            data_list.append(data)
    return data_list
