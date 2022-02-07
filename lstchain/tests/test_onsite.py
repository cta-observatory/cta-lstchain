from pathlib import Path

def test_create_pro_link(tmp_path: Path):
    from lstchain.onsite import create_pro_symlink

    v1 = tmp_path / 'v1'
    v2 = tmp_path / 'v2'
    pro = tmp_path / 'pro'

    v1.mkdir()
    v2.mkdir()

    # test pro does not yet exist
    create_pro_symlink(v1)
    assert pro.exists()
    assert pro.resolve() == v1

    # test pro exists and points to older version
    create_pro_symlink(v2)
    assert pro.exists()
    assert pro.resolve() == v2
