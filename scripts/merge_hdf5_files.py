import argparse
import os
import h5py
import tables

parser = argparse.ArgumentParser(description="Merge all HDF5 files resulting from parallel reconstructions \
 present in a directory. Every dataset in the files must be readable with pandas.")


parser.add_argument('--source-dir', '-d', type=str,
                    dest='srcdir',
                    help='path to the source directory of files',
                    )

parser.add_argument('--outfile', '-o', action='store', type=str,
                    dest='outfile',
                    help='Path of the resulting merged file',
                    default='merge.h5')

args = parser.parse_args()


def get_dataset_keys(filename):
    """
    Return a list of all dataset keys in a HDF5 file

    Parameters
    ----------
    filename: str - path to the HDF5 file

    Returns
    -------
    list of keys
    """
    dataset_keys = []
    def walk(name, obj):
        if type(obj) == h5py._hl.dataset.Dataset:
            dataset_keys.append(name)

    with h5py.File(filename,'r') as file:
        file.visititems(walk)

    return dataset_keys



if __name__ == '__main__':
    file_list = [args.srcdir + '/' + f for f in os.listdir(args.srcdir) if f.endswith('.h5')]

    keys = get_dataset_keys(file_list[0])
    groups = set([k.split('/')[0] for k in keys])

    f1 = tables.open_file(file_list[0])
    merge_file = tables.open_file(args.outfile, 'w')

    nodes = {}
    for g in groups:
        nodes[g] = f1.copy_node('/', name=g, newparent=merge_file.root, newname=g, recursive=True)


    for filename in file_list[1:]:
        file = tables.open_file(filename)

        for ii, node in nodes.items():
            for children in node:
                print(ii)
                print(type(ii))
                print(node)
                print(type(node))
                p = os.path.join(ii, children.name)
                node[children.name].append(file.root[p].read())