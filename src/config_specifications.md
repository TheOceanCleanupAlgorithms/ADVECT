TODO

from run_advector.py
:param eddy_diffusivity: (m^2 / s) controls the scale of each particle's random walk.  0 (default) has no effect.
        Note: since eddy diffusivity parameterizes ocean mechanics at smaller scales than the current files resolve,
            the value chosen should reflect the resolution of the current files.  Further, though eddy diffusivity in
            the real ocean varies widely in space and time, ADVECTOR uses one value everywhere, and the value should be
            selected with this in mind.