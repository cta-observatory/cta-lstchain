from pathlib import Path

DEFAULT_R0_PATH = Path('/fefs/aswg/data/real/R0')


def create_pro_symlink(output_dir):
    '''Create or update the pro symlink to given ``output dir``'''
    output_dir = Path(output_dir).expanduser().resolve()
    pro_dir = output_dir.parent / "pro"

    # remove previous pro link, if it points to an older version
    if pro_dir.exists() and pro_dir.resolve() != output_dir.resolve():
        pro_dir.unlink()

    if not pro_dir.exists():
        pro_dir.symlink_to(output_dir)


def find_r0_subrun(run, sub_run, r0_dir=DEFAULT_R0_PATH):
    '''
    Find the given subrun R0 file (i.e. globbing for the date part)
    '''
    file_list = sorted(r0_dir.rglob(f'LST-1.1.Run{run:05d}.{sub_run:04d}*.fits.fz'))

    if len(file_list) == 0:
        raise IOError(f"Run {run} not found\n")
    else:
        return file_list[0]
