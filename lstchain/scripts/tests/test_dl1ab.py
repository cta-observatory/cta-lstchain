from lstchain.scripts.lstchain_dl1ab import includes_image_modification


def test_image_modifier_checker_no_modifiers():
    config = {}
    assert not includes_image_modification(config)


def test_image_modifier_checker_empty_modifier():
    config = {"image_modifier": {}}
    assert not includes_image_modification(config)


def test_image_modifier_checker_with_modifiers():
    config = {"image_modifier": {"increase_psf": True, "increase_nsb": False}}
    assert includes_image_modification(config)
    config = {"image_modifier": {"increase_nsb": True, "increase_psf": False}}
    assert includes_image_modification(config)
    config = {"image_modifier": {"increase_psf": True, "increase_nsb": True}}
    assert includes_image_modification(config)
    config = {"image_modifier": {"increase_psf": False, "increase_nsb": False}}
    assert not includes_image_modification(config)

