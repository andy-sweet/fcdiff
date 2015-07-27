.. _methods:

Methods
*******

This section describes our mathematical model of differences in functional
connectivity and the methods we use to estimate anomalous connections and
regions in unhealthy patients and groups.


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

First, we describe a model where the anomalous regions are not shared across
the group of patients, though their parameters or effects are.


Generative Model
----------------

Let :math:`R_{nu}` be a Bernoulli random variable indicating that
region :math:`n` of patient :math:`u` is anomalous. :math:`R_{nu}` is drawn
from the distribution

.. math::
    p(r_{nu}; \pi)  = \pi^{r_{nu}} (1 - \pi)^{r_{nu}}

where :math:`\pi \in (0, 1)` is the parameter of a Bernoulli distribution.

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
        \eta^{t_{nmu}} (1 - \eta)^{1 - t_{nmu}} & \mathrm{if} \, r_{nu} \neq r_{mu} = 0
    \end{cases}

where :math:`\delta` is the Dirac delta function and :math:`\eta \in (0, 1)`
is the parameter of a Bernoulli distribution. :math:`T_{nmu}` is
deterministic if the anomalous state of regions :math:`n` and :math:`m` in
patient :math:`u` is the same, and is a Bernoulli random variable with
parameter :math:`\eta` if they are different. This distribution encourages
anomalous networks containing cliques of anomalous nodes, where larger
values of :math:`\eta` allow more edges outside of cliques to be affected.

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
vector of a Multinomial distribution such that
:math:`\gamma_k \in (0, 1) \, \forall k` and
:math:`\sum_{k = -1}^1 \gamma_k = 1`.

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

Let :math:`B_{nmh}` be the random Pearson correlation coefficient between the
fMRI BOLD contrast time series from regions :math:`n` and :math:`m` of healthy
subject :math:`h`. :math:`B_{nmh}` is dependent on the healthy
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

Similarly, let :math:`\tilde{B}_{nmu}` denote the random Pearson correlation
coefficient between the fMRI BOLD contrast time series from regions :math:`n`
and :math:`m` of patient :math:`u`. :math:`\tilde{B}_{nmu}` is dependent on
the connectivity state of the patient, and is drawn from the same
mixture of Normal distributions as the healthy correlations

.. math::
    p(\tilde{b}_{nmh} | \tilde{f}_{nm}; \mu, \sigma)
    =
    \prod_{k = -1}^1
        \mathcal{N}(\tilde{b}_{nmh}; \mu_k, \sigma_k^2)^{\tilde{f}_{nmk}}

except that it is conditional on :math:`\tilde{F}` instead of :math:`F`.

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


Inference Algorithm
-------------------

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

Next, we marginalize over :math:`\tilde{F}_{nmu}`

.. math::
    p(\tilde{b}_{nmu} | f_{nm}, r_{nu}, r_{mu}; \theta)
    &=
    \sum_{\tilde{f}_{nmu}}
        p(\tilde{f}_{nmu} | f_{nm}, r_{nu}, r_{mu}; \eta, \epsilon)
        p(\tilde{b}_{nmu} | \tilde{f}_{nmu}; \mu, \sigma)
    \\
    &=
    \prod_{k = -1}^{1}
        \mathcal{M}_{0k}(\tilde{b}_{nmu}; \theta)^{f_{nmk} r_{nu} r_{mu}}
        \mathcal{M}_{1k}(\tilde{b}_{nmu}; \theta)^{f_{nmk} (1 - r_{nu}) (1 - r_{mu})}
    \\
    & \qquad
        \mathcal{M}_{\neq k}(\tilde{b}_{nmu}; \theta)^{f_{nmk} (r_{nu} (1 - r_{mu}) + (1 - r_{nu}) r_{mu})}

where

.. math::
    \mathcal{M}_{0k}(\tilde{b}_{nmu}; \theta)
    &=
    (1 - \epsilon) \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
    +
    \frac{\epsilon}{2} \sum_{l \neq k} \mathcal{N}(\tilde{b}_{nmu}; \mu_l, \sigma_l^2)
    \\
    \mathcal{M}_{1k}(\tilde{b}_{nmu}; \theta)
    &=
    \epsilon \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
    +
    \frac{1 - \epsilon}{2} \sum_{l \neq k} \mathcal{N}(\tilde{b}_{nmu}; \mu_l, \sigma_l^2)
    \\
    \mathcal{M}_{\neq k}(\tilde{b}_{nmu}; \theta)
    &=
    \tilde{\epsilon} \mathcal{N}(\tilde{b}; \mu_k, \sigma_k^2)
    +
    \frac{1 - \tilde{\epsilon}}{2} \sum_{l \neq k} \mathcal{N}(\tilde{b}_{nmu}; \mu_l, \sigma_l^2).

.. note::
    The marginalization of :math:`T` comes at the expense of coupling
    :math:`\epsilon` and :math:`\eta` with
    :math:`\mu` and :math:`\sigma`.



