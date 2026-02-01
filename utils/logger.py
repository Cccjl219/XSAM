import csv
import os

class CSVLogger(object):
    def __init__(self, args, fieldnames, filename):
        self.fieldnames = fieldnames
        self.filename = filename
        self.first_write = not os.path.exists(filename)
        mode = 'a' if not self.first_write else 'w'
        with open(self.filename, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if self.first_write:  # 如果是第一次写入，则写入表头
                csvfile.write(str(args) + '\n')
                writer.writeheader()
                self.first_write = False  # 标记为已写入

    def save_values(self, *values):
        assert len(values) == len(self.fieldnames), 'The number of values should match the number of field names.'
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            row = {}
            for i, val in enumerate(values):
                row[self.fieldnames[i]] = val

            writer.writerow(row)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
