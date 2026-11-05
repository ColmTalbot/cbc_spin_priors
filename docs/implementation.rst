Implementation
==============

Every prior needs a CDF for ``cdf``, an inverse CDF for ``rescale`` and a
density for ``prob``. None of them use lookup tables. The aligned-spin CDFs
are closed form. The in-plane CDFs are a fixed-order quadrature of closed-form
integrands. The inverses are safeguarded Newton iterations. This page
describes each piece, then how the code is split between the numba, numpy and
array-API paths, and how its accuracy is tested. The notation follows
:doc:`theory`.

Closed-form CDFs
----------------

The dilogarithm
^^^^^^^^^^^^^^^

The aligned-spin CDFs need the real part of the dilogarithm
:math:`\mathrm{Li}_2(z)` for real :math:`z`. ``_spence`` in
``aligned.py`` (numba) and :func:`cbc_spin_priors._xp.spence` (array API)
implement :math:`\mathrm{Li}_2(1 - x)` for :math:`x \geq 0`. They use the
same Cephes rational approximation and argument reductions as
``scipy.special.spence``. For :math:`z > 1`, the real part comes from the
reflection

.. math::

   \Re\,\mathrm{Li}_2(z) = \frac{\pi^2}{3} - \frac{1}{2}\ln^2 z
   - \mathrm{Li}_2(1/z).

``ChiEffQ``
^^^^^^^^^^^

Scale the aligned spins as :math:`x = \chi_1/\amax` and
:math:`y = \chi_2/\amax`. Each has density :math:`f(x) = -\ln|x|/2` on
:math:`[-1, 1]` and CDF :math:`1/2 + g(x)`, with

.. math::

   g(x) = \frac{x (1 - \ln|x|)}{2} \quad (|x| \leq 1), \qquad
   g(x) = \tfrac{1}{2} \quad (x > 1).

The prior is symmetric. With :math:`t = (1 + q)|\chieff|/\amax = |x + qy|`,

.. math::

   F(\chieff | q) = \frac{1}{2} + \mathrm{sign}(\chieff)\, H(t), \qquad
   H(t) = P(0 < x + qy < t) = \int_{-1}^{1} dy\, f(y)\, g(t - qy).

The term :math:`\int f(y) g(-qy) dy` vanishes because :math:`f` is even
and :math:`g` is odd. Substituting :math:`z = t - qy`,

.. math::

   H(t) = -\frac{1}{2q} \int_{t - q}^{t + q} dz\,
   \left(\ln|z - t| - \ln q\right) g(z).

This has a closed-form antiderivative on :math:`|z| \leq 1`
(``_chi_eff_antiderivative``). It is built from
:math:`z\ln|z|`, :math:`\ln|z - t|` and
:math:`t^2\, \Re\,\mathrm{Li}_2(1 - z/t)`. The coefficient of
:math:`\ln|z - t|` vanishes at :math:`z = t`, so that singularity is
removable. The piece with :math:`z > 1`, present when :math:`t + q > 1`, is
elementary. The density is the derivative of the same expression,

.. math::

   p(\chieff | q) = \frac{1 + q}{\amax} \cdot \frac{1}{4q}
   \int dz\, \ln|z|\, \ln\frac{|t - z|}{q},

over :math:`z \in [t - q, \min(t + q, 1)]`. It is evaluated with the
:class:`~cbc_spin_priors.aligned.Chi2` antiderivative below. The Newton inverse and
the generic path's ``prob`` use this form. For numpy inputs, ``prob``
evaluates Callister's piecewise form (see :doc:`theory`) with
``scipy.special.spence``.

``Chi2``
^^^^^^^^

With :math:`\beta = (1 + q)\chieff / q`, the unnormalised density is
:math:`\ln(|\chi_2|/\amax)\,\left(\ln q + \ln|\beta - \chi_2| - \ln\amax\right)`.
Its antiderivative (``_chi2_antiderivative``) follows from

.. math::

   \int \frac{\ln|u|}{1 - u}\, du = \Re\,\mathrm{Li}_2(1 - u),

after integrating by parts. The case :math:`\beta = 0` is separate. The CDF
is :math:`[A(\chi_2) - A(\chi_{2,\min})] / [A(\chi_{2,\max}) - A(\chi_{2,\min})]`,
and the denominator also normalises ``prob``.

The in-plane priors
^^^^^^^^^^^^^^^^^^^

Three pieces of :doc:`theory` are closed form.

- **The disk probability** :math:`G_\infty(a, d)` before truncation
  (``_disk_probability``, ``_nb_disk``). With
  :math:`x = a^2 + c^2 - d^2`, :math:`z = c^2 + d^2 - a^2` and
  :math:`\sigma = \sqrt{x^2 + 4c^2d^2}`, it is
  :math:`\mathrm{log1p}(m)/(2L)`, with

  .. math::

     m = \frac{2a^2}{\sigma + z} \quad (z > 0), \qquad
     m = \frac{\sigma - z}{2c^2} \quad (z \leq 0).

  The two forms are algebraically equal. Each avoids cancellation in its
  range.
- **The truncated part**, where the disk crosses :math:`|v_1| = R_1`, is a
  one-dimensional integral. It is evaluated with 16-point Gauss–Legendre
  quadrature (``N_INNER``) after substituting
  :math:`r^2 = a^2 + d^2 - 2ad\cos\tau`. This removes the square-root
  behaviour at :math:`r = a + d`, so the integrand is smooth on
  :math:`\tau \in [\tau_0, \pi]`, where :math:`r(\tau_0) = R_1`.
- **The arc integral** :math:`J(a, d)` (``_arc_integral``, ``_nb_dgda``) gives
  :math:`\partial G / \partial a = a J / (\pi L_1)`. That is the integrand of
  the densities of :math:`|S|` and :math:`|v_2|`. :math:`J` and the CDF of
  :math:`\phi_{S2}` and its inverse are arctangents, evaluated with
  ``atan2`` so that :math:`\psi_c = \pi` needs no special case.

Quadrature over the secondary spin
----------------------------------

The CDF of :math:`|S|` and the CDF and normalisation of :math:`|v_2|` are
integrals over :math:`|v_2|`. They are computed in the variable
:math:`t = F(|v_2| \,|\, \chi_2) = \ln(1 + |v_2|^2/\chi_2^2)/(2L_2)`, so that
the prior on :math:`|v_2|` becomes the integration measure. For
:math:`d = q^2|v_2|` below :math:`a - R_1`, the disk contains the whole
support and :math:`G = 1`. This contributes :math:`t_\mathrm{lo}`
exactly, so

.. math::

   F(|S|) = t_\mathrm{lo} + \int_{t_\mathrm{lo}}^{t_\mathrm{hi}}
   G\left(a, d(t)\right) dt.

Knots
^^^^^

The integrand is smooth except at four points, which are the knots of the
quadrature (``_Geometry.set_rho``, ``_nb_knots``):

- :math:`t_\mathrm{lo}` and :math:`t_\mathrm{hi}`, at
  :math:`d = \max(0, a - R_1)` and :math:`d = \min(q^2 R_2, a + R_1)`. The
  integrand has square-root behaviour at these points.
- :math:`t_k`, at :math:`d = R_1 - a`. This is where the disk first crosses
  :math:`|v_1| = R_1` and the truncation switches on, so there is a kink.
- :math:`t_a`, at :math:`d = a`. This is where the circle passes through the
  origin. For small :math:`|\chi_1|`, the density of :math:`v_1` has a peak
  of width :math:`c_1` there, so the integrand changes over a width of
  :math:`\epsilon_a = (c_1/q^2)\, p(r_a | \chi_2)` in :math:`t`, with
  :math:`r_a = a/q^2`. If :math:`t_a` is outside
  :math:`[t_\mathrm{lo}, t_\mathrm{hi}]`, it is moved to the nearest end. Its
  width becomes the hypotenuse of :math:`\epsilon_a` and the distance moved.

The interior knots are sorted, giving three segments. Each segment is split
at its midpoint into two halves, and each half is anchored at its knot. That
makes six half segments.

The sinh map
^^^^^^^^^^^^

Each half segment, from anchor :math:`x_0` towards its end over a span
:math:`\ell`, is mapped as

.. math::

   x = x_0 \pm \epsilon\,\sinh(Y s^2), \qquad
   Y = \operatorname{arcsinh}(\ell/\epsilon), \qquad s \in [0, 1].

Then 20-point Gauss–Legendre quadrature (``N_OUTER``) is applied in
:math:`s` (``_half_segment_nodes``, ``_nb_map``). The map does two things.

- The :math:`s^2` makes a square-root endpoint smooth in :math:`s`.
- The sinh clusters nodes within :math:`\epsilon` of the anchor, which
  resolves a feature of width :math:`\epsilon` there.

Knots without a near-singular feature have :math:`\epsilon = \infty`. That
is the pure quadratic map :math:`x = x_0 \pm \ell s^2`. In the code,
:math:`\epsilon` is clipped to :math:`[10^{-14}\ell, 10^3\ell]`.

One CDF evaluation of :math:`|S|` therefore costs
:math:`6 \times 20 = 120` evaluations of :math:`G`, each with 16 inner
nodes. The density uses the same nodes with :math:`\partial G/\partial a`.
For :math:`|v_2|`, the integrand is :math:`\partial G/\partial a`. A partial
integral up to a point inside a half segment rescales the same map to
:math:`s \in [0, s_{\max}]`.

Inverse CDFs
------------

``rescale`` solves :math:`F(x) = u` by Newton's method, safeguarded by
bisection. Each element keeps a bracket :math:`[\mathrm{lo}, \mathrm{hi}]`
that is updated from the sign of the residual. A Newton step is replaced by
bisection when any of these holds:

- it leaves the bracket;
- the derivative is not finite or not positive;
- the residual did not halve since the previous step.

The iteration stops when the residual is below a tolerance or the bracket is
narrower than a relative :math:`10^{-14}`–:math:`10^{-15}`. It runs at most
100 times. The variable and starting point are chosen per prior so that the
function is close to linear:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - prior
     - variable
     - start and tolerance
   * - ``ChiEffQ``
     - :math:`t = (1 + q)|\chieff|/\amax`, solving
       :math:`H(t) = |u - 1/2|`
     - linear near zero, with slope :math:`1 - \ln(q)/2`. Residual
       :math:`10^{-16}`, or stops once a Newton step is below a relative
       :math:`10^{-10}`.
   * - ``Chi2``
     - :math:`\chi_2` on its support
     - linear interpolation of :math:`u`. Residual :math:`10^{-14}` of the
       normalisation.
   * - ``ChiInPlaneTotal``
     - :math:`\mathrm{logit}(|S|/|S|_{\max})`, solving
       :math:`\mathrm{logit} F = \mathrm{logit}\, u`
     - :math:`\mathrm{logit}(u)/2`, bracket :math:`[-700, 40]`. The CDF is a
       power law in both tails, so this is nearly linear. Residual
       :math:`10^{-10}` in logit.
   * - ``ChiInPlane2GivenTotal``
     - the half segment containing :math:`u` is found from the cumulative
       half-segment integrals, then :math:`s` of the sinh map is solved for
     - :math:`\sqrt{\text{fraction}}`. Residual :math:`10^{-13}` of the
       normalisation.
   * - ``PhiS2``
     - closed form
     -

The ``CMIN`` floor
------------------

The in-plane priors clip :math:`|\chi_i|` to
:math:`[\mathrm{CMIN}, \amax(1 - 10^{-10})]`, with
``CMIN`` :math:`= 10^{-10}` (``_spin_constants``, ``_nb_spin``). The lower
floor keeps :math:`L_i = \ln(\amax/|\chi_i|)` and the width
:math:`\epsilon_a \propto c_1` finite. Without it, :math:`\chi_i = 0`
exactly would give infinite logarithms and a zero-width feature. The upper
clip keeps :math:`R_i > 0`.

This changes the prior only for :math:`|\chi_i| < 10^{-10}`. Under the
standard prior, that has probability
:math:`(\epsilon/\amax)(1 + \ln(\amax/\epsilon)) \approx 2 \times 10^{-9}`
for :math:`\epsilon = 10^{-10}`. In that range the in-plane spin is drawn as
if :math:`|\chi_i| = 10^{-10}`. The aligned-spin priors need no floor: their
logarithmic singularities are integrated in closed form.

Numba kernels and the numpy fallback
------------------------------------

For numpy arrays and Python scalars, the inputs are broadcast and flattened.
A compiled scalar kernel then runs over each element. The kernels are
compiled with ``numba.njit(cache=True, error_model="numpy")`` on first use
and cached on disk.

- In ``inplane.py``, the ``_nb_*`` kernels exist only if numba can be
  imported. Without numba, a separate vectorised numpy implementation is used
  (``_Geometry`` and the ``_*_numpy`` functions). Its Newton iterations work
  on the shrinking set of unconverged elements. The module variable
  ``cbc_spin_priors.inplane.BACKEND`` selects the path: ``"numba"`` (the
  default if numba is installed), ``"numpy"``, or ``"array_api"``, which
  sends numpy inputs through the generic path. The functional interface
  (``total_cdf_pdf``, ``total_rescale``, ``secondary_cdf_pdf``,
  ``secondary_rescale``) also takes ``backend=``.
- In ``aligned.py``, the same scalar functions are decorated with
  ``numba.njit`` if numba is available and run as plain Python otherwise.
  The results are the same, but plain Python is much slower.

The in-plane angle prior :class:`~cbc_spin_priors.inplane.PhiS2` and the
conversions :func:`~cbc_spin_priors.inplane.in_plane_from_total` and
:func:`~cbc_spin_priors.inplane.total_from_in_plane` are closed form. They have a
single array-API implementation, which numpy also uses through
``array_api_compat``.

The generic array-API path
--------------------------

Dispatch
^^^^^^^^

:func:`cbc_spin_priors._xp.namespace` chooses the namespace from the
arguments:

- Python scalars and ``None`` are ignored. Objects that ``array_api_compat``
  does not recognise, such as pandas series, count as numpy.
- If every array is numpy, the namespace is numpy, unless the ``xp`` keyword
  passed by bilby's ``xp_wrap`` names another library. In that case ``xp`` is
  used.
- Numpy arrays mixed with arrays of one other library use the other library.
  This handles conditioning variables from priors that only return numpy.
  Two different non-numpy libraries raise ``TypeError``.

A numpy namespace goes to the numba or numpy path. Any other namespace goes
to ``_inplane_xp.py`` or ``_aligned_xp.py``.
:func:`cbc_spin_priors._xp.as_float` broadcasts the inputs to flat float64
arrays on the device of the first array. Arrays that are already in the
namespace are cast with ``astype`` rather than ``asarray``, which would drop
torch's autograd history.

Structure
^^^^^^^^^

The generic modules transcribe the numba kernels element by element, written
only with array-API operations:

- the batch is the leading axis. The six half segments, the 20 outer nodes
  and the 16 inner nodes are trailing axes. One in-plane CDF evaluation
  therefore holds arrays of :math:`6 \times 20 \times 16` values per element;
- there is no boolean-mask indexing or in-place assignment. Every branch is
  evaluated on every element and combined with ``xp.where``. The
  denominators and arguments of the unused branch are made safe, so that
  they do not produce NaN gradients. For example,
  :func:`cbc_spin_priors._xp.acos_clip` has a zero gradient where its
  argument is clipped;
- :func:`cbc_spin_priors._xp.newton_bisect` is the vectorised Newton
  iteration. It evaluates every element on every iteration and freezes
  converged ones with ``xp.where``. It exits early once all elements have
  converged, but only for eager arrays, because ``array_api_compat.is_lazy_array``
  arrays cannot be inspected.

Limitations
^^^^^^^^^^^

- **jit:** the closed-form and quadrature parts, the CDFs and densities,
  trace under ``jax.jit``. This is tested for the CDF of :math:`|S|`. The
  inverse CDFs cannot exit early under tracing, so all 100 iterations are
  unrolled and compilation is slow. About 15 iterations suffice for the
  in-plane priors and ``ChiEffQ``, and about 50 for ``Chi2``. The internal
  rescale functions take ``maxiter``, but the prior classes do not expose
  it yet.
- **Autodiff:** gradients of ``rescale`` go through the Newton iterations,
  not through the implicit function theorem. They are correct (tested as
  :math:`\partial x/\partial u = 1/p(x)`), but the backward pass costs about
  as much as the forward one.
- **Precision:** everything is computed in float64. JAX needs
  ``jax.config.update("jax_enable_x64", True)``.
- **Speed:** on CPU the generic path is several times slower than numba,
  about 15 times for ``ChiInPlaneTotal``.
- **bilby's priors:** some of bilby's own priors, such as ``AlignedSpin`` and
  ``UniformInComponentsMassRatio``, do not accept non-numpy arrays yet. A
  full ``BBHPriorDict`` therefore still needs numpy inputs.

Tested accuracy
---------------

The test suite in ``tests/`` checks the following.

**Distributions.** Samples from every parameterisation agree with uniform
magnitudes and isotropic tilts in KS tests (:math:`p > 10^{-3}`). For the
in-plane priors, this includes the joint distribution against bilby's
standard in-plane priors. Standard samples mapped through the new CDFs are
uniform in the new unit cube.

**ChiEffQ.**

- The CDF agrees with adaptive quadrature of the density to
  :math:`10^{-10}`, including points within :math:`10^{-9}` of the
  breakpoints of the density.
- The density integrates to 1 within :math:`10^{-10}`.
- ``cdf(rescale(u))`` returns :math:`u` within :math:`10^{-13}`, and
  ``rescale(cdf(x))`` returns :math:`x` within :math:`10^{-12}`.

**Chi2.** The CDF agrees with quadrature to :math:`10^{-7}`, and the round
trip returns :math:`u` within :math:`10^{-10}`.

**In-plane priors.** The tests use aligned spins down to :math:`10^{-8}`,
equal masses and the minimum mass ratio.

- The CDF and rescale round trip returns :math:`u` within :math:`10^{-9}`
  for :math:`|S|` and :math:`|v_2|`, and within :math:`10^{-10}` for
  :math:`\phi_{S2}`. The values come back within a relative :math:`10^{-6}`.
- The densities integrate to the CDF differences within :math:`10^{-7}` and
  match histograms.
- The product of the three conditional densities equals the standard density
  times the finite-difference Jacobian within a relative :math:`10^{-5}`.
- Numba and the numpy fallback agree within a relative :math:`10^{-9}` for
  the CDFs and :math:`10^{-7}` for the inverses.

**Generic path.** It agrees with numba within a relative :math:`10^{-10}`
for numpy, array-api-strict, torch and JAX, as installed. The ill-conditioned
edge cases (aligned spins below ``CMIN`` or at :math:`\amax`) agree within
:math:`10^{-7}`. The dilogarithm matches ``scipy.special.spence`` within
:math:`10^{-13}`. Torch gradients of ``rescale`` and ``cdf`` match
:math:`1/p` and :math:`p` within relative :math:`10^{-6}` and
:math:`10^{-8}`, and a gradient with respect to a conditioning variable
matches finite differences.
