from lstchain.onsite import create_symlink_overwrite


def test_create_symlink_overwrite(tmp_path):
    target1 = tmp_path / 'target1'
    target1.open('w').close()

    target2 = tmp_path / 'target2'
    target2.open('w').close()

    # link not yet existing case
    link = tmp_path / 'link'
    create_symlink_overwrite(link, target1)
    assert link.resolve() == target1.resolve()

    # link points to the wrong target, recreate
    create_symlink_overwrite(link, target2)
    assert link.resolve() == target2.resolve()


    # link exists, points already to the target, this should be a no-op
    # but I didn't find a good way to verify that it really is one
    create_symlink_overwrite(link, target2)
    assert link.resolve() == target2.resolve()
