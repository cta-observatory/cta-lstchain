from pathlib import Path

def create_pro_symlink(output_dir):
    '''Create or update the pro symlink to given ``output dir``'''
    output_dir = Path(output_dir).expanduser().resolve()
    pro_dir = output_dir.parent / "pro"

    # remove previous pro link, if it points to an older version
    if pro_dir.exists() and pro_dir.resolve() != output_dir.resolve():
        pro_dir.unlink()

    if not pro_dir.exists():
        pro_dir.symlink_to(output_dir)

