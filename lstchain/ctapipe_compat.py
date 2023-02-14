"""
Module containing workarounds / wrappers for ctapipe code with fixes for lstchain.

The idea is that everything here should be fixed upstream and than an import
like `from lstchain.ctapipe_compat import Foo` can just be replaced by doing
`from ctapipe.<module> import Foo` when upgrading to a ctapipe version containing
the fix.
"""


__all__ = [
]
