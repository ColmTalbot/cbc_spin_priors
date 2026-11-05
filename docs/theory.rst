Parameterisations
=================

All the parameterisations in this package describe one distribution of the
two dimensionless spin vectors. They differ only in the coordinates that the
sampler sees. This page derives the densities in each set of coordinates and
the Jacobians between them. :doc:`implementation` describes how they are
evaluated.

Notation: :math:`m_1 \geq m_2`, :math:`q = m_2 / m_1 \leq 1`, and
:math:`\boldsymbol{\chi}_i = \mathbf{S}_i / m_i^2` are the dimensionless
spins. The aligned component is :math:`\chi_i` (``chi_1``, ``chi_2``), the
component along the orbital angular momentum :math:`\mathbf{L}`. The
in-plane component is the 2D vector :math:`v_i = \chi_{i\perp} e^{i\phi_i}`,
written as a complex number, with magnitude :math:`|v_i| = \chi_{i\perp}`
(``chi_1_in_plane``, ``chi_2_in_plane``).

The standard prior
------------------

The standard prior draws each spin magnitude uniformly up to
:math:`\amax` and each direction isotropically:

.. math::

   a_i \sim U(0, \amax), \quad \cos\theta_i \sim U(-1, 1), \quad
   \phi_i \sim U(0, 2\pi).

Each spin vector then has density
:math:`p(\boldsymbol{\chi}_i) = 1 / (4\pi \amax |\boldsymbol{\chi}_i|^2)` in
the ball :math:`|\boldsymbol{\chi}_i| \leq \amax`. Integrating out the
in-plane components gives the aligned-spin prior

.. math::

   p(\chi_i) = \frac{1}{2\amax} \ln\frac{\amax}{|\chi_i|}, \quad |\chi_i| \leq \amax,

which is bilby's ``AlignedSpin`` with a uniform magnitude. Dividing the joint
density by it gives the in-plane spin given the aligned one,

.. math::

   p(v_i | \chi_i) = \frac{1}{2\pi L_i (\chi_i^2 + |v_i|^2)}, \quad
   |v_i| \leq R_i = \sqrt{\amax^2 - \chi_i^2}, \quad
   L_i = \ln\frac{\amax}{|\chi_i|}.

This depends only on :math:`|v_i|`, so the azimuth is uniform and the
magnitude has density and CDF

.. math::

   p(\chi_{i\perp} | \chi_i) = \frac{\chi_{i\perp}}{L_i (\chi_i^2 + \chi_{i\perp}^2)},
   \qquad
   F(\chi_{i\perp} | \chi_i) = \frac{1}{2L_i} \ln\left(1 + \frac{\chi_{i\perp}^2}{\chi_i^2}\right).

This is bilby's ``ConditionalChiInPlane``.

The two spins are independent, so the overall azimuth of the pair is uniform
and independent of everything else. bilby follows the LALSimulation
convention. It places :math:`v_1` along a reference axis and keeps only the
relative azimuth :math:`\phi_{12}`. The overall orientation is set by
:math:`\phi_{JL}`, the azimuth of :math:`\mathbf{L}` about the total angular
momentum :math:`\mathbf{J}`. When the waveform is generated, the spins are
rotated so that the in-plane part of :math:`\mathbf{J}` is the reference
direction, and only then is :math:`\phi_{JL}` applied. The reference axis of
:math:`v_1` therefore drops out.

Every parameterisation below reproduces this distribution. The ``aligned``
parameterisation samples it directly as :math:`(\chi_1, \chi_2)`, then
:math:`(\chi_{1\perp}, \chi_{2\perp}, \phi_{12})`. The ``spherical``
parameterisation samples :math:`(a_i, \theta_i)` and :math:`\phi_{12}`.

Why change coordinates
----------------------

A sampler works best when each well-measured combination of parameters is a
coordinate of its own. Then the posterior is narrow along that coordinate and
close to the prior along the others. The individual spin components are
rarely measured well. A few combinations are.

- The **effective spin**

  .. math::

     \chieff = \frac{\chi_1 + q\chi_2}{1 + q}

  is the mass-weighted aligned spin that enters the inspiral phase at leading
  order, and it is conserved under orbit-averaged precession. It is usually
  the best-measured spin parameter, and it is strongly correlated with the
  mass ratio. With the mass ratio and :math:`\chieff` measured, the
  posterior on :math:`(\chi_1, \chi_2)` is a thin strip along a line of
  constant :math:`\chieff`. In :math:`(\chieff, \chi_2)` that strip is
  aligned with an axis.

- The **total in-plane spin** controls precession. The orbital plane
  precesses about :math:`\mathbf{J} = \mathbf{L} + \mathbf{S}_1 + \mathbf{S}_2`.
  The opening angle :math:`\beta` of the precession cone satisfies

  .. math::

     \tan\beta = \frac{|\mathbf{J}_\perp|}{|\mathbf{L}| + S_{1\parallel} + S_{2\parallel}},
     \qquad
     \mathbf{J}_\perp = m_1^2 (v_1 + q^2 v_2).

  The data constrain the size of the precession-induced modulation, and so
  :math:`|v_1 + q^2 v_2|`. The split between the two spins and the angle
  between them are much harder to measure. In the standard coordinates,
  a measurement of :math:`|\mathbf{J}_\perp|` is a curved surface in
  :math:`(\chi_{1\perp}, \chi_{2\perp}, \phi_{12})`. In the coordinates
  below, it is a constraint on one coordinate.

Effective-spin aligned parameterisation
---------------------------------------

The ``effective`` parameterisations sample :math:`\chieff` given :math:`q`
(:class:`~cbc_spin_priors.aligned.ChiEffQ`), then :math:`\chi_2` given
:math:`(q, \chieff)` (:class:`~cbc_spin_priors.aligned.Chi2`), and set
:math:`\chi_1` with a conditional delta function
(:class:`~cbc_spin_priors.aligned.Chi1`).

Jacobian
^^^^^^^^

At fixed :math:`q`,

.. math::

   \chi_1 = (1 + q)\chieff - q\chi_2, \qquad
   \left|\frac{\partial(\chi_1, \chi_2)}{\partial(\chieff, \chi_2)}\right| = 1 + q,

so the standard prior in the new coordinates is

.. math::

   p(\chieff, \chi_2 | q) = (1 + q)\, p(\chi_1)\, p(\chi_2)
   = \frac{1 + q}{4\amax^2} \ln\frac{\amax}{|\chi_1|} \ln\frac{\amax}{|\chi_2|},

on the region where :math:`|\chi_1|, |\chi_2| \leq \amax`.

:math:`\chieff` given :math:`q`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The marginal is

.. math::

   p(\chieff | q) = \frac{1 + q}{4\amax^2} \int d\chi_2\,
   \ln\frac{\amax}{|(1 + q)\chieff - q\chi_2|} \ln\frac{\amax}{|\chi_2|}.

It is symmetric in :math:`\chieff`, with support
:math:`|\chieff| \leq \amax`. `Callister (2021)
<https://arxiv.org/abs/2104.09508>`_ gives it in closed form. With
:math:`x = |\chieff|`,

.. math::

   P = \frac{\amax}{1 + q}, \qquad s = \frac{q\amax}{1 + q}, \qquad
   \Delta = P - s = \frac{(1 - q)\amax}{1 + q},

it is

.. math::

   p(\chieff | q) = \begin{cases}
       \dfrac{(1 + q)(2 - \ln q)}{2\amax} & x = 0, \\[2ex]
       \dfrac{1}{4P}\left[4 - 2\ln q - \ln\left|\dfrac{x^2}{s^2} - 1\right|
       + \dfrac{x}{s}\left(\ln\left|\dfrac{x - s}{x + s}\right|
       + \mathrm{Li}_2\left(-\dfrac{s}{x}\right)
       - \Re\,\mathrm{Li}_2\left(\dfrac{s}{x}\right)\right)\right]
       & 0 < x < \Delta, \\[2ex]
       \dfrac{1}{4Ps}\Big[2(\amax - x) + \Delta\ln q
       + (x - s)\ln\left|\dfrac{x}{s} - 1\right|
       + (x - P)\ln\left|\dfrac{x}{P} - 1\right| & \\
       \qquad + x\ln\dfrac{P}{x}\left(\ln\left|\dfrac{x}{P} - 1\right| - \ln q\right)
       + x\left(\mathrm{Li}_2\left(1 - \dfrac{P}{x}\right)
       - \Re\,\mathrm{Li}_2\left(\dfrac{s}{x}\right)\right)\Big]
       & \Delta \leq x < \amax, \\[2ex]
       0 & x \geq \amax.
   \end{cases}

Here :math:`P` and :math:`s` are the largest values of :math:`|\chieff|` that
the primary and the secondary alone can produce. For :math:`x < \Delta` the
two aligned spins may have opposite signs. Above :math:`\Delta` they must have
the same sign. Callister separates further cases by comparing :math:`x` with
:math:`s`. Taking the absolute values and real parts above covers them with
one expression on each side of :math:`\Delta`. At :math:`q = 1` the first
range is empty.

The value at :math:`\chieff = 0` follows directly from the integral:

.. math::

   p(0 | q) = \frac{1 + q}{4\amax^2}\int_{-\amax}^{\amax} d\chi_2\,
   \ln\frac{\amax}{q|\chi_2|}\ln\frac{\amax}{|\chi_2|}
   = \frac{(1 + q)(2 - \ln q)}{2\amax}.

:math:`\chi_2` given :math:`(q, \chieff)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dividing the joint density by the marginal gives

.. math::

   p(\chi_2 | q, \chieff) = \frac{1 + q}{4\amax^2\, p(\chieff | q)}
   \ln\frac{|\chi_2|}{\amax} \ln\frac{|(1 + q)\chieff - q\chi_2|}{\amax}.

Both logarithms are negative on the support, so their product is positive.
The support is the part of :math:`[-\amax, \amax]` where
:math:`|(1 + q)\chieff - q\chi_2| \leq \amax`. For :math:`\chieff \geq 0`,
this is

.. math::

   \max\left(\frac{(1 + q)\chieff - \amax}{q}, -\amax\right) \leq \chi_2 \leq \amax,

and it is mirrored for :math:`\chieff < 0`. The density has integrable
logarithmic singularities at :math:`\chi_2 = 0` and
:math:`\chi_2 = (1 + q)\chieff / q`, where :math:`\chi_1 = 0`. Its
normalisation is :math:`4\amax^2 p(\chieff | q) / (1 + q)`, which gives an
independent check of the closed form above.

:math:`\chi_1` as a conditional delta function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`(q, \chieff, \chi_2)`, :math:`\chi_1 = (1 + q)\chieff - q\chi_2`
exactly. :class:`~cbc_spin_priors.aligned.Chi1` is a conditional delta function at
this value, so the prior density of a sample is
:math:`p(\chieff | q)\, p(\chi_2 | q, \chieff)`. This is
:math:`(1 + q)\, p(\chi_1)\, p(\chi_2)`, the standard prior times the
Jacobian.

Total in-plane spin parameterisation
------------------------------------

The ``_perp`` parameterisations keep the aligned spins of ``aligned`` or
``effective``. They replace :math:`(\chi_{1\perp}, \chi_{2\perp}, \phi_{12})`
with:

1. ``chi_in_plane_total``: :math:`|S|`, the magnitude of the total in-plane
   spin

   .. math::

      S = \frac{v_1 + q^2 v_2}{1 + q^2},

   (:class:`~cbc_spin_priors.inplane.ChiInPlaneTotal`);
2. ``chi_2_in_plane``: :math:`|v_2|` given :math:`|S|`
   (:class:`~cbc_spin_priors.inplane.ChiInPlane2GivenTotal`);
3. ``phi_s2``: the angle :math:`\phi_{S2} \in (-\pi, \pi]` of :math:`v_2`
   relative to :math:`S` (:class:`~cbc_spin_priors.inplane.PhiS2`);
4. ``chi_1_in_plane`` and ``phi_12`` as conditional delta functions
   (:class:`~cbc_spin_priors.inplane.ChiInPlane1FromTotal`,
   :class:`~cbc_spin_priors.inplane.Phi12FromTotal`), through

   .. math::

      v_1 = (1 + q^2) S - q^2 v_2.

All three sampled priors are conditioned on :math:`\chi_1`, :math:`\chi_2`
and :math:`q`.

:math:`S` is the in-plane part of :math:`\mathbf{J}` in units of the primary
mass: :math:`\mathbf{J}_\perp = m_1^2 v_1 + m_2^2 v_2 = m_1^2 (1 + q^2) S`.
The factor :math:`1 + q^2` makes :math:`|S| \leq \amax`. The angle
:math:`\phi_{S2}` measures :math:`v_2` from the direction of
:math:`\mathbf{J}_\perp`. This is the direction that the waveform convention
uses as its reference before it applies :math:`\phi_{JL}`, so
:math:`\phi_{JL}` keeps its uniform prior and takes the role of the azimuth
of :math:`S`.

Throughout this section,

.. math::

   a = (1 + q^2)|S|, \qquad d = q^2|v_2|, \qquad
   c_i = |\chi_i|, \qquad R_i = \sqrt{\amax^2 - \chi_i^2}, \qquad
   L_i = \ln\frac{\amax}{c_i}.

With :math:`S` along the real axis, :math:`v_1 = a - d e^{i\phi_{S2}}`. So
:math:`v_1` lies on the circle of radius :math:`a` centred a distance
:math:`d` from the origin, and

.. math::

   \chi_{1\perp}^2 = |v_1|^2 = a^2 + d^2 - 2ad\cos\phi_{S2}, \qquad
   \phi_{12} = \phi_{S2} - \arg\left(a - d e^{i\phi_{S2}}\right).

Jacobians
^^^^^^^^^

At fixed :math:`q`, the map :math:`(v_1, v_2) \to (S, v_2)` is linear. It
scales areas in the :math:`v_1` plane by :math:`(1 + q^2)^2`. In polar
coordinates :math:`S = |S| e^{i\alpha}` and
:math:`v_2 = |v_2| e^{i(\alpha + \phi_{S2})}`, so

.. math::

   d^2v_1\, d^2v_2 = (1 + q^2)^2\, |S|\, |v_2|\;
   d|S|\, d\alpha\, d|v_2|\, d\phi_{S2}.

The standard prior :math:`p(v_1 | \chi_1)\, p(v_2 | \chi_2)` depends on the
vectors only through :math:`|v_1|` and :math:`|v_2|`. Neither depends on
:math:`\alpha`, so :math:`\alpha` is uniform and independent. Integrating it
out gives the joint density of the sampled coordinates:

.. math::

   p(|S|, |v_2|, \phi_{S2} | \chi_1, \chi_2, q) =
   \frac{(1 + q^2)^2\, |S|\, |v_2|}
        {2\pi L_1 L_2\, (c_1^2 + a^2 + d^2 - 2ad\cos\phi_{S2})\,(c_2^2 + |v_2|^2)},

on :math:`|v_1| \leq R_1` and :math:`|v_2| \leq R_2`. Equivalently, the
Jacobian to the standard in-plane coordinates is

.. math::

   \left|\frac{\partial(\chi_{1\perp}, \chi_{2\perp}, \phi_{12})}
              {\partial(|S|, |v_2|, \phi_{S2})}\right|
   = \frac{(1 + q^2)^2 |S|}{\chi_{1\perp}},

and
:math:`p(|S|, |v_2|, \phi_{S2}) = p(\chi_{1\perp} | \chi_1)\, p(\chi_{2\perp} | \chi_2) / (2\pi)`
times this Jacobian.

The conditional densities below are the factors of this joint density, taken
in the sampling order.

:math:`\phi_{S2}` given :math:`(|S|, |v_2|)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At fixed :math:`|S|` and :math:`|v_2|`, the density in :math:`\phi_{S2}` is
the standard density of :math:`v_1` along the circle:

.. math::

   p(\phi_{S2} | \ldots) = \frac{1}{2 J(a, d)}
   \frac{1}{c_1^2 + a^2 + d^2 - 2ad\cos\phi_{S2}},
   \qquad |\phi_{S2}| \leq \psi_c.

Here :math:`\psi_c` is the half-width of the arc that lies inside
:math:`|v_1| \leq R_1`:

.. math::

   \cos\psi_c = \frac{a^2 + d^2 - R_1^2}{2ad},

clipped to :math:`[-1, 1]`, so :math:`\psi_c = \pi` when the whole circle is
inside. With :math:`A_\mp = c_1^2 + (a \mp d)^2`, the arc integral is

.. math::

   J(a, d) = \int_0^{\psi_c} \frac{d\psi}{c_1^2 + a^2 + d^2 - 2ad\cos\psi}
   = \frac{2}{\sqrt{A_- A_+}}
     \arctan\left(\sqrt{\frac{A_+}{A_-}} \tan\frac{\psi_c}{2}\right).

The CDF is

.. math::

   F(\phi_{S2}) = \frac{1}{2}\left[1 +
     \frac{\arctan\left(k\tan(\phi_{S2}/2)\right)}
          {\arctan\left(k\tan(\psi_c/2)\right)}\right],
   \qquad k = \sqrt{A_+ / A_-},

and its inverse is closed form as well.

:math:`|v_2|` given :math:`|S|`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integrating :math:`\phi_{S2}` over the arc gives

.. math::

   p(|v_2| \,|\, |S|) \propto
   \frac{|v_2|}{c_2^2 + |v_2|^2}\, J(a, q^2|v_2|).

The support is the set of :math:`|v_2| \leq R_2` for which the circle reaches
the disk :math:`|v_1| \leq R_1`, that is, :math:`|a - d| \leq R_1`:

.. math::

   \max(0, a - R_1) \leq q^2|v_2| \leq \min(q^2 R_2, a + R_1).

:math:`|S|`
^^^^^^^^^^^

Integrating again over :math:`|v_2|`,

.. math::

   p(|S|) = \frac{(1 + q^2)^2 |S|}{\pi L_1}
   \int d|v_2|\; p(|v_2| \,|\, \chi_2)\; J\left((1 + q^2)|S|, q^2|v_2|\right),

on :math:`0 \leq |S| \leq (R_1 + q^2 R_2)/(1 + q^2)`. The CDF is easiest to
write as a probability. :math:`|S| \leq \rho` is the same as :math:`v_1`
lying in the disk of radius :math:`(1 + q^2)\rho` centred at
:math:`-q^2 v_2`. So

.. math::

   F(\rho) = \int d|v_2|\; p(|v_2| \,|\, \chi_2)\;
   G\left((1 + q^2)\rho,\, q^2|v_2|\right),

where :math:`G(a, d)` is the probability, under :math:`p(v_1 | \chi_1)`, of
the disk of radius :math:`a` centred a distance :math:`d` from the origin.
Its derivative is the integral of the density along the boundary circle,
:math:`\partial G / \partial a = a J(a, d) / (\pi L_1)`, which gives the
density above.

Without the truncation at :math:`R_1`, :math:`G` is closed form. In polar
coordinates about the centre of the disk, the angular integral gives
:math:`2\pi/\sqrt{(c^2 + r^2 + d^2)^2 - 4r^2d^2}`. The radial integral is
then elementary:

.. math::

   G_\infty(a, d) = \frac{1}{2L_1}
   \ln\frac{a^2 + c_1^2 - d^2 + \sqrt{(a^2 + c_1^2 - d^2)^2 + 4c_1^2 d^2}}{2c_1^2}.

When the disk crosses :math:`|v_1| = R_1`, the probability outside
:math:`R_1` is subtracted. For each radius :math:`r > R_1`, the fraction of
the circle :math:`|v_1| = r` inside the disk is
:math:`\arccos\left[(r^2 + d^2 - a^2)/(2rd)\right] / \pi`, so

.. math::

   G(a, d) = G_\infty(a, d) - \int_{\max(R_1, |a - d|)}^{a + d}
   dr\, \frac{r}{L_1(c_1^2 + r^2)}\,
   \frac{1}{\pi}\arccos\frac{r^2 + d^2 - a^2}{2rd}

for :math:`a - R_1 \leq d \leq a + R_1`. Outside this range the disk either
contains the whole support (:math:`G = 1`) or misses it (:math:`G = 0`).

:math:`\chi_{1\perp}` and :math:`\phi_{12}` as conditional delta functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`(|S|, |v_2|, \phi_{S2}, q)`, the primary in-plane spin is fixed
by :math:`v_1 = (1 + q^2) S - q^2 v_2`.
:func:`~cbc_spin_priors.inplane.in_plane_from_total` returns
:math:`\chi_{1\perp} = |v_1|` and :math:`\phi_{12} \in [0, 2\pi)`, and
:func:`~cbc_spin_priors.inplane.total_from_in_plane` inverts it. The support
conditions above guarantee :math:`\chi_{1\perp} \leq R_1`, so both spin
magnitudes are at most :math:`\amax`.

Small aligned spins
^^^^^^^^^^^^^^^^^^^

As :math:`\chi_i \to 0`, :math:`L_i` diverges and :math:`\chi_{i\perp}` given
:math:`\chi_i` becomes log-uniform between about :math:`|\chi_i|` and
:math:`R_i`. The density of :math:`v_1` then has a sharp peak of width
:math:`|\chi_1|` at the origin. The circle in the :math:`v_1` plane passes
through the origin when :math:`d = a`, so the integrands above vary on that
scale there. The densities stay finite for :math:`\chi_i \neq 0`. The
implementation floors :math:`|\chi_i|` at :math:`10^{-10}`; see
:doc:`implementation`.
