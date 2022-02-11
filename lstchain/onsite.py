from pathlib import Path
import tempfile


def create_symlink_overwrite(link, target):
    '''
    Create a symlink from link to target, replacing an existing link atomically.
    '''
    if not link.exists():
        link.symlink_to(target)
        return

    if link.resolve() == target.resolve():
        # nothing to do
        return

    # create the symlink in a tempfile, then replace the original one
    # in one step to avoid race conditions
    tmp = Path(tempfile.mktemp(prefix='tmp_symlink'))
    tmp.symlink_to(target)
    tmp.replace(link)


def create_pro_symlink(output_dir):
    output_dir = Path(output_dir)
    pro_dir = output_dir / '../pro'
    create_symlink_overwrite(pro_dir, output_dir)
