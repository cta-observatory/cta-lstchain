from lstchain.scripts.lstchain_dl1ab import image_modifier_checker


def test_image_modifier_checker_no_modifiers():
    config = {}
    assert not image_modifier_checker(config)


def test_image_modifier_checker_empty_modifier():
    config = {"image_modifier": {}}
    assert not image_modifier_checker(config)


def test_image_modifier_checker_with_modifiers():
    config = {"image_modifier": {"increase_psf": True, "increase_nsb": False}}
    assert image_modifier_checker(config)
    config = {"image_modifier": {"increase_nsb": True, "increase_psf": False}}
    assert image_modifier_checker(config)
    config = {"image_modifier": {"increase_psf": True, "increase_nsb": True}}
    assert image_modifier_checker(config)
    config = {"image_modifier": {"increase_psf": False, "increase_nsb": False}}
    assert not image_modifier_checker(config)

