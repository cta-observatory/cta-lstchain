# T. Vuillaume, 20/06/2019
## slice a list of lines into files with fixed number of lines
## this script is used to generate smaller lists of files to send parallel jobs

import argparse
import os

parser = argparse.ArgumentParser(description='Slice a file in smaller files')
parser.add_argument('input_file', type=str,
                   help='Path to the input file')

parser.add_argument('--number-line', '-n', dest='n', action='store',
                   default=10,
                   help='Number of lines per output file')

parser.add_argument('--output-dir', '-o', dest='output_dir', action='store',
                   default=os.getcwd(),
                   help='Output directory')

args = parser.parse_args()

input_file = args.input_file
n = int(args.n)

i = 0


print("Dividing lines from file {} into set of {} lines in {} directory".format(
    input_file,
    n,
    args.output_dir,)
)
    

os.makedirs(args.output_dir, exist_ok=True)

with open(input_file) as fi:
    l = fi.readlines()
    for i in range(len(l)//n + int(len(l)%n>0) ):
        output_file = args.output_dir + "/{}.list".format(i)
        with open(output_file, "w+") as fo:
            for line in l[i*n:min((i+1)*n, len(l))]:
                fo.write(line)
    i+=1
    print("{} files generated".format(len(l)//n+1))
