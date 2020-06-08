from eda.eda import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--i", required=True, type=str, help="原始数据的输入文件目录")
ap.add_argument("--o", required=False, type=str, help="增强数据后的输出文件目录")
ap.add_argument("--n", required=False, type=int, help="每条原始语句增强的语句数", default=10)
ap.add_argument("--alpha", required=False, type=float, help="每条语句中将会被改变的单词数占比")
args = ap.parse_args()

o = None
if args.o:
    o = args.o
else:
    from os.path import dirname, basename, join
    o = join(dirname(args.i), 'eda_' + basename(args.i))

n = 10
if args.n:
    n = args.n

alpha = 0.2
if args.alpha:
    alpha = args.alpha


def gen_eda(train_orig, output_file, alpha, num_aug=10):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()

    print("利用easy data agument")
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = parts[0]
        sentence = parts[1]
        aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(label + "\t" + aug_sentence + '\n')

    writer.close()
    print("agument finish")
    print(output_file)


if __name__ == "__main__":
    gen_eda(args.i, o, alpha=alpha, num_aug=n)