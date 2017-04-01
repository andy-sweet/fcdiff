.. _methods:

Methods
*******

This section describes our mathematical models of differences in functional
connectivity and the methods we use to estimate anomalous connections and
regions in unhealthy patients and groups. All these models are defined in
:mod:`fcdiff.model`.


Nomenclature
============

A brief list of letter and symbols used for mathematical notation.

- :math:`N` : number of regions.

- :math:`H` : number of healthy subjects.

- :math:`U` : number of unhealthy patients.

- :math:`R` : anomalous region indicator.

- :math:`T` : anomalous connection indicator.

- :math:`F` : functional connectivity of a subject.

- :math:`\tilde{F}` : functional connectivity of a patient.

- :math:`B` : correlation of a subject.

- :math:`\tilde{B}` : correlation of a patient.


Individual Anomalous Regions
============================

First, we describe the individual anomalous region (IAR) model, in which
the anomalous regions are not shared across the group of patients, though
their parameters or effects are. This model is defined in
:class:`fcdiff.model.UnsharedRegionModel`.


Generative Model
----------------

.. currentmodule:: fcdiff.model

If you are familiar with graphical models, then look at
:numref:`fig_graphical_model` for a summary of the generative model.

.. _fig_graphical_model:

.. figure:: images/graphical_model_all.*
    :width: 100 %
    :align: center

    The graph that represents the conditonal dependences between hidden random
    variables (unshaded circles), observed random variables (shaded
    circles) and unknown fixed parameters (rounded rectangles) in our
    model. The sharp rectangles are plates that represent the number of
    times a variable or parameter is repeated.
    The right side of the figure shows the relationship between :math:`R`
    and :math:`T`.

Now we go into more detail on the distributions of these random variables and
their parameters.

Let :math:`R_{nu}` be a Bernoulli random variable indicating that
region :math:`n` of patient :math:`u` is anomalous. :math:`R_{nu}` is drawn
from the distribution

.. math::
    p(r_{nu}; \pi)  = (1 - \pi)^{1 - r_{nu}} \pi^{r_{nu}}

where :math:`\pi \in (0, 1)` is the parameter of a Bernoulli distribution.
You can sample from this distribution with
:meth:`UnsharedRegionModel.sample_R`.

Let :math:`T_{nmu}` be a Bernoulli random variable indicating that the
connection between regions :math:`n` and :math:`m` of patient :math:`u` is
anomalous. :math:`T_{nmu}` is dependent on the anomalous state of the regions
at either end of the connection, and is drawn from the distribution

.. math::
    p(t_{nmu} | r_{nu}, r_{mu}; \eta)
    \begin{cases}
        \delta(t_{nmu}) & \mathrm{if} \, r_{nu} = r_{mu} = 0
        \\
        \delta(1 - t_{nmu}) & \mathrm{if} \, r_{nu} = r_{mu} = 1
        \\
        (1 - \eta)^{1 - t_{nmu}} \eta^{t_{nmu}} & \mathrm{if} \, r_{nu} \neq r_{mu}
    \end{cases}

where :math:`\delta` is the Dirac delta function and :math:`\eta \in (0, 1)`
is the parameter of a Bernoulli distribution. :math:`T_{nmu}` is
deterministic if the anomalous state of regions :math:`n` and :math:`m` in
patient :math:`u` is the same, and is a Bernoulli random variable with
parameter :math:`\eta` if they are different.
This distribution encourages anomalous networks containing cliques of
anomalous nodes, where larger values of :math:`\eta` allow more edges
outside of cliques to be affected.
You can sample from this distribution with
:meth:`UnsharedRegionModel.sample_T`.

Let :math:`F_{nm}` be a multinomial random variable indicating the state of
healthy connectivity between regions :math:`n` and :math:`m`.
We use three states of connectivity:

- :math:`f_{nm-1} = 1` denotes negative connectivity;
- :math:`f_{nm0} = 1` denotes no connectivity;
- :math:`f_{nm1} = 1` denotes positive connectivity.

Exactly one component of :math:`f_{nm}` must be equal to one. :math:`F_{nm}`
is drawn from the distribution

.. math::
    p(f_{nm}; \gamma)
    =
    \prod_{k = -1}^1
        \gamma_k^{f_{nmk}}

where :math:`\gamma = (\gamma_{-1}, \gamma_0, \gamma_{1})` is the parameter
vector of a multinomial distribution such that
:math:`\gamma_k \in (0, 1) \, \forall k` and
:math:`\sum_{k = -1}^1 \gamma_k = 1`.
You can sample from this distribution with
:meth:`UnsharedRegionModel.sample_F`.

Let :math:`\tilde{F}_{nmu}` be a multinomial random variable indicating the
state of connectivity between regions :math:`n` and :math:`m` of
patient :math:`u`. :math:`\tilde{F}_{nmu}` is dependent on :math:`T_{nmu}`,
the anomalous state of the connection between regions :math:`n` and :math:`m`
of patient :math:`u`, and on :math:`F_{nm}`, the healthy
connectivity state between regions :math:`n` and :math:`m`.
:math:`\tilde{F}_{nmu}` is drawn from the distribution

.. math::
    p(\tilde{f}_{nmu} | f_{nm}, t_{nmu}; \epsilon)
    =
    \begin{cases}
        (1 - \epsilon)^{f_{nm}^\top \tilde{f}_{nmu}}
        \left( \frac{\epsilon}{2} \right)^{1 - f_{nm}^\top \tilde{f}_{nmu}}
        &
        \mathrm{if} \, t_{nmu} = 0
        \\
        \epsilon^{f_{nm}^\top \tilde{f}_{nmu}}
        \left( \frac{1 - \epsilon}{2} \right)^{1 - f_{nm}^\top \tilde{f}_{nmu}}
        &
        \mathrm{if} \, t_{nmu} = 1
    \end{cases}

where :math:`\epsilon \in (0, 1)` is the parameter of a Bernoulli
distribution. If the connection between regions :math:`n` and :math:`m` of
patient :math:`u` is anomalous, the connectivity state is perturbed from the
healthy template with high probability :math:`1 - \epsilon`. Conversely, if
the connection is typical, the connectivity state is perturbed with
small probability :math:`\epsilon`.
You can sample from this distribution with
:meth:`UnsharedRegionModel.sample_F_tilde`.

Let :math:`B_{nmh}` be the random Pearson correlation coefficient between the
fMRI BOLD contrast time series from regions :math:`n` and :math:`m` of healthy
subject :math:`h`. This is dependent on the healthy
connectivity state, and is drawn from a mixture of Normal distributions

.. math::
    p(b_{nmh} | f_{nm}; \mu, \sigma)
    =
    \prod_{k = -1}^1
        \mathcal{N}(b_{nmh}; \mu_k, \sigma_k^2)^{f_{nmk}}

where :math:`\mu = (\mu_{-1}, \mu_0, \mu_{1})`,
:math:`\sigma = (\sigma_{-1}, \sigma_0, \sigma_{1})` and
:math:`\mathcal{N}(\cdot; \mu_k, \sigma_k^2)` is a Normal distribution
with mean :math:`\mu_k` and variance :math:`\sigma_k^2`.
You can sample from this distribution with
:meth:`UnsharedRegionModel.sample_B`.

Similarly, let :math:`\tilde{B}_{nmu}` denote the random Pearson correlation
coefficient between the fMRI BOLD contrast time series from regions :math:`n`
and :math:`m` of patient :math:`u`.
This is dependent on the connectivity state of the patient, and is drawn from
the same mixture of Normal distributions as the healthy correlations

.. math::
    p(\tilde{b}_{nmh} | \tilde{f}_{nm}; \mu, \sigma)
    =
    \prod_{k = -1}^1
        \mathcal{N}(\tilde{b}_{nmh}; \mu_k, \sigma_k^2)^{\tilde{f}_{nmk}}

except that it is conditional on :math:`\tilde{F}` instead of :math:`F`.
You can sample from this distribution with
:meth:`UnsharedRegionModel.sample_B_tilde`.

We assume independence between all healthy subjects and patients,
independence between healthy connections and independence between regions,
and thus obtain the full joint distribution

.. math::
    p&(f, b, r, t, \tilde{f}, \tilde{b}; \theta)
    =
    p(f; \gamma)
    p(b | f; \mu, \sigma)
    p(r; \pi)
    p(t | r; \eta)
    p(\tilde{f} | f, t; \epsilon)
    p(\tilde{b} | \tilde{f}; \mu, \sigma)
    \\
    =
    &\left(
        \prod_{n = 1}^N
        \prod_{m > n}
            p(f_{nm}; \gamma)
        \prod_{h = 1}^H
            p(b_{nmh} | f_{nm}; \mu, \sigma)
    \right)
    \\
    &\left(
        \prod_{n = 1}^N
        \prod_{u = 1}^U
            p(r_{nu}; \pi)
            \prod_{m > n}
                p(t_{nmu} | r_{nu}, r_{mu}; \eta)
                p(\tilde{f}_{nmu} | f_{nm}, t_{nmu}; \epsilon)
                p(\tilde{b} | \tilde{f}_{nmu}; \mu, \sigma)
    \right)

where :math:`\theta = (\pi, \gamma, \eta, \epsilon, \mu, \sigma)`.


Exact Inference
---------------

In :cite:`Sweet:2013a` three inference algorithms are discussed. Here,
we only present the algorithm that gave the best estimation performance
on synthetic data experiments and computational efficiency. Similarly,
only this algorithm is available in the toolbox.

Our goal is to infer the posterior probability distribution
:math:`p(r_{nu} | \tilde{b}, b; \theta)` for all regions
:math:`n \in \{1, ..., N\}` and in all patients
:math:`u \in \{1, ..., U\}`.
This requires marginalizing out all latent random variables in the model
to compute the partition function :math:`p(\tilde{b}, b; \theta)`.

First, note that we can easily sum over :math:`T_{nmu}`

.. math::
    p(\tilde{f}_{nmu} | f_{nm}, r_{nu}, r_{mu}; \eta, \epsilon)
    &=
    \sum_{t_{nmu}}
        p(t_{nmu} | r_{nu}, r_{mu}; \eta)
        p(\tilde{f}_{nmu} | f_{nm}, t_{nmu}; \epsilon)
    \\
    &=
    \begin{cases}
        (1 - \epsilon)^{f_{nm}^\top \tilde{f}_{nmu}}
        \left( \frac{\epsilon}{2} \right)^{1 - f_{nm}^\top \tilde{f}_{nmu}}
        &
        \mathrm{if} \, r_{nu} = r_{mu} = 0
        \\
        \epsilon^{f_{nm}^\top \tilde{f}_{nmu}}
        \left( \frac{1 - \epsilon}{2} \right)^{1 - f_{nm}^\top \tilde{f}_{nmu}}
        &
        \mathrm{if} \, r_{nu} = r_{mu} = 1
        \\
        \tilde{\epsilon}^{f_{nm}^\top \tilde{f}_{nmu}}
        \left( \frac{1 - \tilde{\epsilon}}{2} \right)^{1 - f_{nm}^\top \tilde{f}_{nmu}}
        &
        \mathrm{if} \, r_{nu} \neq r_{mu}
    \end{cases}

where :math:`\tilde{\epsilon} = \eta \epsilon + (1 - \eta)(1 - \epsilon)`.

.. note::
    The marginalization of :math:`T` comes at the expense of coupling
    :math:`\epsilon` with :math:`\eta`.

Next, we marginalize out :math:`\tilde{F}_{nmu}`

.. math::
    p(\tilde{b}_{nmu} | f_{nm}, r_{nu}, r_{mu}; \theta)
    &=
    \sum_{\tilde{f}_{nmu}}
        p(\tilde{f}_{nmu} | f_{nm}, r_{nu}, r_{mu}; \eta, \epsilon)
        p(\tilde{b}_{nmu} | \tilde{f}_{nmu}; \mu, \sigma)
    \\
    &=
    \sum_{k=-1}^{1}
        \mathcal{N}(\tilde{b}_{nmu}; \mu_{k}, \sigma_{k}^2)
    \\
    &\qquad
        \left(
            (1 - \epsilon)^{f_{nmk}}
            \left(
                \frac{\epsilon}{2}
            \right)^{1 - f_{nmk}}
        \right)^{(1 - r_{nu}) (1 - r_{mu})}
    \\
    &\qquad
        \left(
            \epsilon^{f_{nmk}}
            \left(
                \frac{1 - \epsilon}{2}
            \right)^{1 - f_{nmk}}
        \right)^{r_{nu} r_{mu}}
    \\
    &\qquad
        \left(
            \tilde{\epsilon}^{f_{nmk}}
            \left(
                \frac{1 - \tilde{\epsilon}}{2}
            \right)^{1 - f_{nmk}}
        \right)^{r_{nu} (1 - r_{mu}) + (1 - r_{nu}) r_{mu}}
    \\
    &=
    \prod_{k = -1}^{1}
        \bigg(
            \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)^{r_{nu} r_{mu}}
            \\
            &\qquad
            \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)^{(1 - r_{nu}) (1 - r_{mu})}
            \\
            &\qquad
            \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)^{(r_{nu} (1 - r_{mu}) + (1 - r_{nu}) r_{mu})}
        \bigg)^{f_{nmk}}

where

.. math::
    \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)
    &=
    (1 - \epsilon) \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
    +
    \frac{\epsilon}{2} \sum_{l \neq k} \mathcal{N}(\tilde{b}_{nmu}; \mu_l, \sigma_l^2)
    \\
    \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)
    &=
    \epsilon \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
    +
    \frac{1 - \epsilon}{2} \sum_{l \neq k} \mathcal{N}(\tilde{b}_{nmu}; \mu_l, \sigma_l^2)
    \\
    \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
    &=
    \tilde{\epsilon} \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
    +
    \frac{1 - \tilde{\epsilon}}{2} \sum_{l \neq k} \mathcal{N}(\tilde{b}_{nmu}; \mu_l, \sigma_l^2).

.. note::
    The marginalization of :math:`T` comes at the expense of coupling
    :math:`\epsilon` and :math:`\eta` with
    :math:`\mu` and :math:`\sigma`.

As :math:`\epsilon` is assumed to be small :math:`\mathcal{M}_{k0}` is
dominated by the likelihood of the correlation being drawn from the
:math:`k^\mathrm{th}` Normal distribution, whereas :math:`\mathcal{M}_{k1}`
is dominated by the likelihoods of the correlation being drawn from the
other Normal distributions. Finally, :math:`\mathcal{M}_{k \neq}` is an
interpolation between the other two terms that tends to
:math:`\mathcal{M}_{k0}` as :math:`\eta \to 0` and tends to
:math:`\mathcal{M}_{k1}` as :math:`\eta \to 1`.

Next, we could marginalize out :math:`F_{nm}`, but this would
complicate the form enough to make inference difficult and would also couple
:math:`\gamma` with :math:`(\epsilon, \eta, \mu, \sigma)`.

.. .. math::
..     p(b_{nm}, \tilde{b}_{nmu} | r_{nu}, r_{mu}; \gamma, \eta, \epsilon, \mu, \sigma)
..     &=
..     \sum_{f_{nm}}
..         p(f_{nm}, b_{nm}, \tilde{b}_{nmu} | r_{nu}, r_{mu}; \gamma, \eta, \epsilon, \mu, \sigma)
..     \\
..     &=
..     \sum_{f_{nm}}
..         p(f_{nm}; \gamma)
..         \prod_{h=1}^H
..             p(b_{nmh} | f_{nm}; \mu, \sigma)
..         p(\tilde{b}_{nmu} | f_{nm}, r_{nu}, r_{mu}; \eta, \epsilon, \mu, \sigma)
..     \\
..     &=
..     \left(
..         \sum_{k=-1}^1
..             \gamma_k
..             \prod_{h=1}^H
..                 \mathcal{N}(b_{nmh}; \mu_k, \sigma_k^2)
..             \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)
..     \right)^{(1 - r_{nu}) (1 - r_{mu})}
..     \\
..     &\quad
..     \left(
..         \sum_{k=-1}^1
..             \gamma_k
..             \prod_{h=1}^H
..                 \mathcal{N}(b_{nmh}; \mu_k, \sigma_k^2)
..             \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)
..     \right)^{r_{nu} r_{mu}}
..     \\
..     &\quad
..     \left(
..         \sum_{k=-1}^1
..             \gamma_k
..             \prod_{h=1}^H
..                 \mathcal{N}(b_{nmh}; \mu_k, \sigma_k^2)
..             \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
..     \right)^{r_{nu} (1 - r_{mu}) + (1 - r_{nu}) r_{mu}}


Finally, note that there is no way to analytically marginalize over :math:`R`,
because of the pair wise conditional dependence between :math:`\tilde{F}` and
:math:`R`. Furthermore, it is very computationally expensive to perform
the brute force summation which would require a sum over :math:`2^N` terms
- one for each value the binary :math:`N` vector can take on.

Now we can construct the joint probability over all the remaining random
variables

.. math::
    p(f, b, r, \tilde{b}; \theta)
    &=
    p(f; \gamma)
    p(b | f; \mu, \sigma)
    p(r; \pi)
    p(\tilde{b} | f, r; \theta)
    \\
    &=
    \left(
        \prod_{n=1}^N
        \prod_{m > n}
            p(f_{nm}; \gamma)
            \prod_{h=1}^H
                p(b_{nmh} | f_{nm}; \mu, \sigma)
    \right)
    \\
    &\quad
    \left(
        \prod_{u=1}^U
        \prod_{n=1}^N
            p(r_{nu}; \pi)
            \prod_{m > n}
                p(\tilde{b}_{nmu} | f_{nm}, r_{nu}, r_{mu}; \eta, \epsilon, \mu, \sigma)
    \right).


Variational Inference
---------------------

To perform inference without marginalizing out the remaining hidden random
variables, :math:`F` and :math:`\tilde{F}`, we use a variational mean field
approximation of their posterior.
The factorization takes the form

.. math::
    p(f, r | b, \tilde{b})
    &\approx
    q(f, r)
    =
    q_F(f) q_R(r)
    \\
    &=
    \left(
        \prod_{n = 1}^{N}
        \prod_{m > n}
        \prod_{k = -1}^{1}
            q_{F_{nmk}}^{f_{nmk}}
    \right)
    \left(
        \prod_{n = 1}^N
        \prod_{u = 1}^U
            q_{R_{nu0}}^{1 - r_{nu}} q_{R_{nu1}}^{r_{nu}}
    \right).

Computing the posterior :math:`p(r | b, \tilde{b}; \theta)` also requires
an estimate of :math:`\theta`.

To estimate both :math:`\theta` and the variational factors, we minimize the
variational free energy

.. math::
    \mathcal{E}(q, \theta; b, \tilde{b})
    &=
    - \mathbb{E}_q \left[ \log p(f, b, r, \tilde{b}; \theta) \right]
    + \mathbb{E}_q \left[ \log q(f, r) \right]
    \\
    &=
    - \mathbb{E}_{q_F} \left[ \log p(f; \gamma) \right]
    - \mathbb{E}_{q_F} \left[ \log p(b | f; \mu, \sigma) \right]
    - \mathbb{E}_{q_R} \left[ \log p(r; \pi) \right]
    \\
    &\qquad
    - \mathbb{E}_{q_F q_R} \left[
        \log p(\tilde{b} | f, r; \epsilon, \eta, \mu, \sigma)
    \right]
    + \mathbb{E}_{q_R} \left[ \log q_R(r) \right]
    + \mathbb{E}_{q_F} \left[ \log q_F(f) \right]

where

.. math::
    \mathbb{E}_{q_F} \left[ \log p(f; \gamma) \right]
    &=
    \sum_{n = 1}^N
    \sum_{m > n}
    \sum_{k = -1}^{1}
        q_{F_{nmk}} \log \gamma_k
    \\
    \mathbb{E}_{q_R} \left[ \log p(r; \pi) \right]
    &=
    \sum_{n = 1}^N
    \sum_{u = 1}^U
        q_{R_{nu0}} \log (1 - \pi)
        +
        q_{R_{nu1}} \log \pi
    \\
    \mathbb{E}_{q_F} \left[ \log p(b | f; \mu, \sigma) \right]
    &=
    \sum_{n = 1}^N
    \sum_{m > n}
    \sum_{k = -1}^{1}
        q_{F_{nmk}}
        \sum_{h = 1}^H
            \log \mathcal{N}(b_{nmh}; \mu_k, \sigma_k^2)
    \\
    \mathbb{E}_{q_F q_R} \left[
        \log p(\tilde{b} | f, r; \epsilon, \eta, \mu, \sigma)
    \right]
    &=
    \sum_{n = 1}^N
    \sum_{m > n}
    \sum_{k = -1}^{1}
        q_{F_{nmk}}
        \sum_{u = 1}^U
        \bigg(
            q_{R_{nu0}} q_{R_{mu0}}
                \log \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)
    \\
    &\qquad
            +
            q_{R_{nu1}} q_{R_{mu1}}
                \log \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)
    \\
    &\qquad
            +
            \left( q_{R_{nu0}} q_{R_{mu1}} + q_{R_{nu1}} q_{R_{mu0}} \right)
                \log \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
        \bigg)
    \\
    \mathbb{E}_{q_F} \left[ \log q(f) \right]
    &=
    \sum_{n = 1}^N
    \sum_{m > n}
    \sum_{k = -1}^{1}
        q_{F_{nmk}} \log q_{F_{nmk}}
    \\
    \mathbb{E}_{q_R} \left[ \log q(r) \right]
    &=
    \sum_{n = 1}^N
    \sum_{u = 1}^U
        q_{R_{nu0}} \log q_{R_{nu0}}
        +
        q_{R_{nu1}} \log q_{R_{nu1}}.

The algorithm for performing this minimization is

.. math::

    \begin{array}{lll}
        \text{Line} & \text{Operation}
        \\ \hline
        1 &
        e \gets \mathcal{E}(q, \theta, b, \tilde{b})
        \\
        2 &
        \text{for } s = 1 \dots S
        \\
        3 & \quad
            q_F^* \gets \mathcal{U}_{q_F}(q_R, \theta, b, \tilde{b})
        \\
        4 & \quad
            q_R^* \gets \mathcal{U}_{q_R}(q_R,q_F^*, \theta, b, \tilde{b})
        \\
        5 & \quad
            \theta^* \gets \mathcal{U}_{\theta}(q^*, \theta, b, \tilde{b})
        \\
        6 & \quad
            e^* \gets \mathcal{E}(q^*, \theta^*, b, \tilde{b})
        \\
        7 & \quad
            \text{if } (e - e^*) / e < \xi
        \\
        8 & \quad \quad
                \text{break}
        \\
        9 & \quad
            q \gets q^*, \, \theta \gets \theta^*, \, e \gets e^*
    \end{array}

where :math:`\xi` is the relative tolerance used to detect convergence before
the maximum number of iteration steps :math:`S`.

The :math:`\mathcal{U}_{q_F}` function is determined by the following update
equation

.. math::
    \log q_{F_{nmk}}^*
    &=
    \log \gamma_k
    +
    \sum_{h=1}^H
        \log \mathcal{N}(b_{nmh}; \mu_k, \sigma_k)
    \\
    &\quad
    +
    \sum_{u=1}^U
        q_{R_{nu0}} q_{R_{mu0}} \log \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)
        +
        q_{R_{nu1}} q_{R_{mu1}} \log \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)
    \\
    &\qquad
        +
        \left(
            q_{R_{nu1}} q_{R_{mu0}}
            +
            q_{R_{nu0}} q_{R_{mu1}}
        \right)
        \log \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
    + \text{const.}

where :math:`\text{const.}` is used to imply that :math:`\log q_{F_{nmk}}^*`
is further normalized to ensure that :math:`\sum_{k=-1}^1 q_{F_{nmk}}^* = 1`.

.. note::
    We optimize w.r.t. :math:`\log q` to ensure positivity of :math:`q`.

The :math:`\mathcal{U}_{q_R}` function is determined by the following update
equations

.. math::
    \log q_{R_{nu0}}^*
    &=
    \log(1 - \pi)
    \\
    &\,
    +
    \sum_{m \neq n}
    \sum_{k=-1}^{1}
        q_{F_{nmk}}
        \left(
            q_{R_{mu0}} \log \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)
            +
            q_{R_{mu1}} \log \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
        \right)
    +
    \text{const.}

.. math::
    \log q_{R_{nu1}}^*
    &=
    \log \pi
    \\
    &\,
    +
    \sum_{m \neq n}
    \sum_{k=-1}^{1}
        q_{F_{nmk}}
        \left(
            q_{R_{mu1}} \log \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)
            +
            q_{R_{mu0}} \log \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
        \right)
    +
    \text{const.}

where :math:`\text{const.}` is used to imply that :math:`\log q_{R_{nu0}}^*`
and :math:`\log q_{R_{nu1}}^*` are further normalized to ensure that
:math:`q_{R_{nu0}}^* + q_{R_{nu1}}^* = 1`.

.. note::
    Here, we estimate each component of the variational parameters and then
    project the solution onto the solution space of probability
    distributions.
    Instead, we could incorporate the equality constraints into the linear
    system of equations. However, note that means we would have to optimize
    w.r.t. :math:`q` instead of :math:`\log q` and thus would need to
    explicitly handle the positivity constraints.

The :math:`\mathcal{U}_{\theta}` function is determined by the following
update equations

.. math::
    \pi^*
    =
    \frac{1}{UN}
    \sum_{u=1}^U
    \sum_{n=1}^N
        q_{R_{nu1}}

.. math::
    \gamma_k^*
    =
    \sum_{n=1}^N
    \sum_{m > n}
        q_{F_{nmk}}
    +
    \text{const.}

as well as the multivariable minimization of :math:`\mathcal{E}` with
respect to :math:`(\mu, \sigma, \epsilon, \eta)`.
This minimization can be performed using a variety of iterative descent
methods.
By default, a trust region reflective Gauss-Newton method is used.
In order to use a descent method, we need the following derivatives.

The derivative of the energy w.r.t. :math:`\mu_j` is

.. math::
    \frac{\partial \mathcal{E}}{\partial \mu_j}
    &=
    -
    \frac{\partial}{\partial \mu_j} \left(
        \mathbb{E}_{q_F} \left[
            \log p(b | f; \mu, \sigma)
        \right]
        +
        \mathbb{E}_{q_F q_R} \left[
            \log p(\tilde{b} | f, r; \theta)
        \right]
    \right)
    \\
    &=
    -
    \sum_{n=1}^N
    \sum_{m > n} \bigg(
        q_{F_{nmj}}
        \sum_{h=1}^H
            \frac{\partial}{\partial \mu_j} \left(
                \log \mathcal{N}(b_{nmh}; \mu_j, \sigma_j^2)
            \right)
    \\
    &\quad
        +
        \sum_{k=-1}^1
        q_{F_{nmk}} \bigg(
            \sum_{u=1}^U
                q_{R_{nu0}} q_{R_{mu0}}
                \frac{\partial}{\partial \mu_j} \left(
                    \log \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)
                \right)
    \\
    &\qquad
                +
                q_{R_{nu1}} q_{R_{mu1}}
                \frac{\partial}{\partial \mu_j} \left(
                    \log \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)
                \right)
    \\
    &\qquad
                +
                \left(
                    q_{R_{nu0}} q_{R_{mu1}}
                    +
                    q_{R_{nu1}} q_{R_{mu0}}
                \right)
                \frac{\partial}{\partial \mu_j} \left(
                    \log \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
                \right)
        \bigg)
    \bigg)

where

.. math::
    \frac{\partial}{\partial \mu_j}
    \left(
        \log \mathcal{M}_{k0}(b; \theta)
    \right)
    =
    \mathcal{M}_{k0}(b; \theta)^{-1}
    (1 - \epsilon)^{\delta(j, k)}
    \left( \frac{\epsilon}{2} \right)^{1 - \delta(j, k)}
    \frac{\partial}{\partial \mu_j}
    \left(
        \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)

.. math::
    \frac{\partial}{\partial \mu_j} \left(
        \log \mathcal{M}_{k1}(b; \theta)
    \right)
    =
    \mathcal{M}_{k1}(b; \theta)^{-1}
    \epsilon^{\delta(j, k)}
    \left( \frac{1 - \epsilon}{2} \right)^{1 - \delta(j, k)}
    \frac{\partial}{\partial \mu_j} \left(
        \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)

.. math::
    \frac{\partial}{\partial \mu_j} \left(
        \log \mathcal{M}_{k \neq}(b; \theta)
    \right)
    =
    \mathcal{M}_{k \neq}(b; \theta)^{-1}
    \tilde{\epsilon}^{\delta(j, k)}
    \left( \frac{1 - \tilde{\epsilon}}{2} \right)^{1 - \delta(j, k)}
    \frac{\partial}{\partial \mu_j} \left(
        \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)

and

.. math::
    \frac{\partial}{\partial \mu_j} \left(
        \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)
    =
    \mathcal{N}(b; \mu_j, \sigma_j^2)
    \frac{\partial}{\partial \mu_j} \left(
        \log \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)

.. math::
    \frac{\partial}{\partial \mu_j} \left(
        \log \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)
    =
    \sigma_j^{-2}
    (b - \mu_j)

The derivative of the energy w.r.t. :math:`\sigma_j^2` is

.. math::
    \frac{\partial \mathcal{E}}{\partial \sigma_j^2}
    &=
    -
    \frac{\partial}{\partial \sigma_j^2} \left(
        \mathbb{E}_{q_F} \left[
            \log p(b | f; \mu, \sigma)
        \right]
        +
        \mathbb{E}_{q_F q_R} \left[
            \log p(\tilde{b} | f, r; \theta)
        \right]
    \right)
    \\
    &=
    -
    \sum_{n=1}^N
    \sum_{m > n} \bigg(
        q_{F_{nmj}}
        \sum_{h=1}^H
            \frac{\partial}{\partial \sigma_j^2} \left(
                \log \mathcal{N}(b_{nmh}; \mu_j^2, \sigma_j^2)
            \right)
    \\
    &\quad
        +
        \sum_{k=-1}^1
        q_{F_{nmk}} \bigg(
            \sum_{u=1}^U
                q_{R_{nu0}} q_{R_{mu0}}
                \frac{\partial}{\partial \sigma_j^2} \left(
                    \log \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)
                \right)
    \\
    &\qquad
                +
                q_{R_{nu1}} q_{R_{mu1}}
                \frac{\partial}{\partial \sigma_j^2} \left(
                    \log \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)
                \right)
    \\
    &\qquad
                +
                \left(
                    q_{R_{nu0}} q_{R_{mu1}}
                    +
                    q_{R_{nu1}} q_{R_{mu0}}
                \right)
                \frac{\partial}{\partial \sigma_j^2} \left(
                    \log \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
                \right)
        \bigg)
    \bigg)

where

.. math::
    \frac{\partial}{\partial \sigma_j^2} \left(
        \log \mathcal{M}_{k0}(b; \theta)
    \right)
    =
    \mathcal{M}_{k0}(b; \theta)^{-1}
    (1 - \epsilon)^{\delta(j, k)}
    \left( \frac{\epsilon}{2} \right)^{1 - \delta(j, k)}
    \frac{\partial}{\partial \sigma_j^2} \left(
        \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)

.. math::
    \frac{\partial}{\partial \sigma_j^2} \left(
        \log \mathcal{M}_{k1}(b; \theta)
    \right)
    =
    \mathcal{M}_{k1}(b; \theta)^{-1}
    \epsilon^{\delta(j, k)}
    \left( \frac{1 - \epsilon}{2} \right)^{1 - \delta(j, k)}
    \frac{\partial}{\partial \sigma_j^2} \left(
        \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)

.. math::
    \frac{\partial}{\partial \sigma_j^2} \left(
        \log \mathcal{M}_{k \neq}(b; \theta)
    \right)
    =
    \mathcal{M}_{k \neq}(b; \theta)^{-1}
    \tilde{\epsilon}^{\delta(j, k)}
    \left( \frac{1 - \tilde{\epsilon}}{2} \right)^{1 - \delta(j, k)}
    \frac{\partial}{\partial \sigma_j^2} \left(
        \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)

and

.. math::
    \frac{\partial}{\partial \sigma_j^2} \left(
        \log \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)
    =
    (2 \sigma_j^2)^{-1} \left(
        (b - \mu_j)^2 - \sigma_j^2
    \right)

.. math::
    \frac{\partial}{\partial \sigma_j^2} \left(
        \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)
    =
    \mathcal{N}(b; \mu_j, \sigma_j^2)
    \frac{\partial}{\partial \sigma_j^2} \left(
        \log \mathcal{N}(b; \mu_j, \sigma_j^2)
    \right)

The derivative of the energy w.r.t. :math:`\eta` is

.. math::
    \frac{\partial \mathcal{E}}{\partial \eta}
    &=
    -
    \frac{\partial}{\partial \eta} \left(
        \mathbb{E}_{q_F q_R} \left[
            \log p(\tilde{b} | f, r; \theta)
        \right]
    \right)
    \\
    &=
    -
    \sum_{n=1}^N
    \sum_{m > n}
    \sum_{k = -1}^1
        q_{F_{nmk}}
        \sum_{u=1}^U
            \left(
                q_{R_{nu0}} q_{R_{mu1}}
                +
                q_{R_{nu1}} q_{R_{mu0}}
            \right)
            \frac{\partial}{\partial \eta} \left(
                \log \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
            \right)

where

.. math::
    \frac{\partial}{\partial \eta} \left(
        \log \mathcal{M}_{k \neq}(\tilde{b}; \theta)
    \right)
    =
    \mathcal{M}_{k \neq}(\tilde{b}; \theta)^{-1}
    \left(
        (2 \epsilon - 1) \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
        -
        \frac{(2 \epsilon - 1)}{2}
        \sum_{l \neq k}
            \mathcal{N}(\tilde{b}; \mu_l, \sigma_l)
    \right)

The derivative of the energy w.r.t. :math:`\epsilon` is

.. math::
    \frac{\partial \mathcal{E}}{\partial \epsilon}
    &=
    -
    \frac{\partial}{\partial \epsilon} \left(
        \mathbb{E}_{q_F q_R} \left[
            \log p(\tilde{b} | f, r; \theta)
        \right]
    \right)
    \\
    &=
    -
    \sum_{n=1}^N
    \sum_{m > n}
    \sum_{k = -1}^1
        q_{F_{nmk}} \bigg(
        \sum_{u=1}^U
            q_{R_{nu0}} q_{R_{mu0}}
            \frac{\partial}{\partial \epsilon} \left(
                \log \mathcal{M}_{k0}(\tilde{b}_{nmu}; \theta)
            \right)
    \\
    &\qquad
            +
            q_{R_{nu1}} q_{R_{mu1}}
            \frac{\partial}{\partial \epsilon} \left(
                \log \mathcal{M}_{k1}(\tilde{b}_{nmu}; \theta)
            \right)
    \\
    &\qquad
            +
            \left(
                q_{R_{nu0}} q_{R_{mu1}}
                +
                q_{R_{nu1}} q_{R_{mu0}}
            \right)
            \frac{\partial}{\partial \epsilon} \left(
                \log \mathcal{M}_{k \neq}(\tilde{b}_{nmu}; \theta)
            \right)
        \bigg)

where

.. math::
    \frac{\partial}{\partial \epsilon} \left(
        \log \mathcal{M}_{k0}(\tilde{b}; \theta)
    \right)
    =
    \mathcal{M}_{k0}(\tilde{b}; \theta)^{-1}
    \left(
        \frac{1}{2}
        \sum_{l \neq k}
            \mathcal{N}(\tilde{b}; \mu_l, \sigma_l^2)
        -
        \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
    \right)

.. math::
    \frac{\partial}{\partial \epsilon} \left(
        \log \mathcal{M}_{k1}(\tilde{b}; \theta)
    \right)
    =
    \mathcal{M}_{k1}(\tilde{b}; \theta)^{-1}
    \left(
        \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
        -
        \frac{1}{2}
        \sum_{l \neq k}
            \mathcal{N}(\tilde{b}; \mu_l, \sigma_l^2)
    \right)

.. math::
    \frac{\partial}{\partial \epsilon} \left(
        \log \mathcal{M}_{k \neq}(\tilde{b}; \theta)
    \right)
    =
    \mathcal{M}_{k \neq}(\tilde{b}; \theta)^{-1}
    \left(
        (2 \eta - 1) \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
        -
        \frac{(2 \eta - 1)}{2}
        \sum_{l \neq k}
            \mathcal{N}(\tilde{b}; \mu_l, \sigma_l^2)
    \right)
