from pathlib import Path
import tempfile


def create_symlink_overwrite(link, target):
    '''
    Create a symlink from link to target, replacing an existing link atomically.
    '''
    target = target.resolve()

    if not link.exists():
        link.symlink_to(target)
        return

    if link.resolve() == target:
        # nothing to do
        return

    # create the symlink in a tempfile, then replace the original one
    # in one step to avoid race conditions
    tmp = tempfile.NamedTemporaryFile(prefix='tmp_symlink_', delete=True)
    tmp.close()
    tmp = Path(tmp.name)
    tmp.symlink_to(target)
    tmp.replace(link)


def create_pro_symlink(output_dir):
    output_dir = Path(output_dir)
    pro_dir = output_dir / '../pro'
    create_symlink_overwrite(pro_dir, output_dir)
