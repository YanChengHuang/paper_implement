import numpy as np
import xlrd

def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)
    booksheet_pos = workbook.sheet_by_index(0)
    nrows_pos = booksheet_pos.nrows
    seq = []
    label = []
    for i in range(nrows_pos):
        seq.append(booksheet_pos.row_values(i)[0])
        label.append(booksheet_pos.row_values(i)[1])
    return seq, np.array(label).astype(int)

def seq_to01_to0123(filename):

    seq, label = read_seq_label(filename)

    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq_start = 0
        for j in range(seq_len):
            seq_0123[i, j] = int(one_seq[j - seq_start])
            if j < seq_start:
                seq_01[i, j, :] = 0.25
            else:
                try:
                    seq_01[i, j, int(one_seq[j - seq_start])] = 1
                except:
                    seq_01[i, j, :] = 0.25
    return seq_01, seq_0123, label

def load_data(filename):

    seq01, _, label = seq_to01_to0123(filename)
    return seq01, label

if __name__ == '__main__':
    filename = 'data/toydata.xlsx'
    load_data(filename)




