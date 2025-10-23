//==============================================================================
// ifacconf.typ 2023-11-17 Alexander Von Moll
// Template for IFAC meeting papers
//
// Adapted from ifacconf.tex by Juan a. de la Puente
//==============================================================================


#import "@preview/abiding-ifacconf:0.2.0": *
#import "@preview/physica:0.9.5": *
#import "@preview/lovelace:0.3.0": *

#show: ifacconf-rules
#show: ifacconf.with(
  title: [#smallcaps([Markov chain Monte Carlo methods\ for Path Integral])\ Thermodynamics and energy gaps of the anharmonic oscillator],
  
  authors: (
    (
      name: "M. Barbieri",
      email: "m.barbieri20@studenti.unipi.it",
      affiliation: 1,
    ),
    (
      name: "M. Dell'Anna",
      email: "m.dellanna2@studenti.unipi.it",
      affiliation: 1,
    ),
  ),
  affiliations: (
    (
      organization: "University of Pisa",
      address: [Largo Pontecorvo 3, I-56127 Pisa, Italy\ ],
    ),
  ),
  abstract: [
The aim of this report is to study the low temperature behavior of an harmonic oscillator with $x^4$ perturbation, focusing in particular on its thermodynamic properties and first few energy gaps.
Single site, single cluster and multi-cluster update algorithms are presented and compared.
  ],
)

#let cal(it) = math.class("normal", box({
  show math.equation: set text(font: "Garamond-Math", stylistic-set: 3)
  $#math.cal(it)$
}) + h(0pt))

#let otimes={math.times.circle}
#set list(body-indent: 0.3em,indent: 0.1em,spacing: 0.5em)
#set enum(body-indent: 0.7em,indent: 0.1em,spacing: 0.5em)
#set math.cases(gap: 1em)
#set table(
  stroke: (x,y) => (
    x: none,
    top: if y == 0 or y == 1 {0.6pt},
    bottom: { 0.6pt },
  ),
  align: (x, y) => if y == 0 {center} else if x == 0 { right } else { left },
  row-gutter: (1.5pt,0pt),
  column-gutter: 0pt,
)
#show table.cell.where(y: 0): strong
#show figure.caption.where(kind: figure): (cont) => {
  block(
    inset: (x: 10pt, y: 0pt),
    outset: 0pt,
    {
    cont
    h(1fr)
    v(-10pt)
  })
}
#show figure.caption.where(kind: table): (cont) => {
  align(
    center,
    cont
  )
}
#show figure.where(kind: table): set figure(supplement: [Table])
#let pm={math.plus.minus}
#let simeq={math.tilde.eq}
#let boring_proof(proof) = {
  parbreak()
  box(baseline: -2pt, line(length: 5%, stroke: 0.3pt))
  h(1fr)
  text(size: 8pt)[Proof]
  h(1fr)
  box(baseline: -2pt, line(length: 80%, stroke: 0.3pt))
  v(-4pt)
  text(size: 8pt, proof)
  v(-5pt)
  line(length: 100%, stroke: 0.3pt)
  v(-2pt)
}
#let tableFromCSV(filename, caption: none, label: none, column-gutter: auto, cell-inset: auto, ..args) = {
  let csvfile = csv("example.csv")
  let csvfile-data = csvfile.slice(1).map( row=>{
    row.map( cell=>{
      eval(cell, mode: "markup")
    })
  })

  set table.cell(inset: cell-inset)
  
  figure(
    table(
      columns: csvfile.first().len(),
      column-gutter: column-gutter,
      table.header( ..csvfile.first().map( title => { eval(title, mode: "markup") } ) ),
      ..csvfile-data.flatten(),
      ..args
    ),
    caption: caption,
  )
}

= Introduction
The system of interest of this report is a single quantum harmonic oscillator, with an anharmonic $x^4$ perturbation.
We focus on thermodynamical properties and energy gaps of such system in the so called Gibbs thermal state, which is a statistical mixture of Hamiltonian eigenstates so that the density matrix $hat(rho)$ corresponds to a Boltzmann distribution:

$ cases(
  hat(H)   = 1/(2 m) hat(p)^2 + (m omega^2) / 2 hat(x)^2 + g/4 hat(x)^4",",
  hat(rho) = 1/Z(beta) dot exp[-beta hat(H)]","
) $
where $Z(beta)$ is the partition function.

The discussion is organized as follows:
- in @simulations we briefly summarize the procedures, choices and conventions used for the simulations;
- in @data_anal we present the statistical and fitting methods used to analyze the numerical simulations;
- in @results we show and comment on the obtained results.


All used codes are available at our repository on GitHub @RepoGH.

= Methods of investigation
<simulations>
== Units
Throughout the document, we set the Boltzmann constant $k_"B" = 1$.
Also, energy is measured in units of the harmonic oscillator gap, therefore $hbar omega = 1$.
Finally, length are expressed in units of harmonic oscillator characteristic length, hence $hbar/(m omega) = 1$

== Path integral and discretization
In order to investigate the wanted properties, we rely on Monte Carlo simulation to obtain the values of correlators with the Path Integral technique.
Standard arguments imply that given the system's (Euclidian) Hamiltonian density $cal(H)$, the respective Euclidian action $S_E$, and an observable $hat(O)$ depending on the position operator only, the following relations hold:
$ S_E [x(tau)](beta) = integral cal(H) [x(tau)] dd(tau), $
and
$ angle.l O angle.r &= tr[O dot rho] = \
  & =1/Z(beta) dot #h(1em) integral_#move($script(x(0)=x(beta))$, dx: -2.5em, dy: 0.3em) #h(-2.25em) cal(D)
  [x(tau)] thick O(x(tau)) thick exp[-S_E [x(tau)] ], quad $
which is the path integral representation of an expected value.
It is clear how the term $1/Z(beta) exp[- S_E]$ can be interpreted as a probability density function; we can discretize the integrals introducing a time step $a = beta/N$, getting:
$ P(x_0, ..., x_(N-1) = x_0) = 1/Z(beta) dot exp[-a sum_(i=0)^(N-1) cal(H) [x_i]], $
which is to be sampled for our investigation.

To this end, we build three different Markov Chains, briefly described in @single_site and @cluster; each one will produce, at each iteration, a new path $(x_0,...,x_(N-1))$, which will be referred to as _state_ of the Markov chain.
We point out that the algorithms are of the Metropolis–Hastings type, and they are only different in how the trial state is chosen.

== Single site update
<single_site>
With this first Markov chain, the initial and trial state only differ by one point of the discretized path.
Given its index $i^star$, and the current state $v = (x_1,...,x_(i^star),...,x_(N-1))$, the trial state becomes
$ w = (x_0,...,x_(i^star) + delta,...,x_(N-1)) $ were the innovation $delta$ must be extracted form a probability density function
$g(delta thin | thin v, i^star)$.
Finally the trial state can be accepted with probability
$ P_("acc") = min(1,g(-delta thin | thin w, i^star)/g(delta thin | thin v, i^star) dot P(w)/P(v)) $
Calling $D^2$ the discrete Laplacian, we observe:
$ #h(-1em) cases(
    display(P(w)/P(v) = exp[-(alpha_v delta - beta_v)^2  + beta_v^2 - a g x_(i^star) delta^3 - (a g)/4 delta^4])"," ,
    display(alpha_v = sqrt(a/2) dot sqrt(2/a^2 +1 + 3 g x_(i^star)^2))"," ,
    beta_v = 1/2 thin a thin alpha_v^(-1) [g x_(i^star)^3 + x_(i^star) - D^2 x_(i^star)]"."
  ) $
We point out that the two coefficients $alpha_v$, $beta_v$ defined above actually depend on initial state and the site that could be updated,
but we will only index them with the former for notation clarity.\
At this point, the better $g(delta thin | thin v, i^star) \/ g(-delta thin |thin w, i^star)$ resembles $P(v) \/ P(w)$, the closer the acceptance probability will be to one.

We also take into account that we need to sample $g(delta thin | thin v)$ for every simulation update, therefore a good choice of this distribution is
the gaussian in @metropolis_gaussian, which returns a decent acceptance probability and is easy to sample via the Box–Müller algorithm:
$ g(delta thin | thin v , i^star) = sqrt(2 / pi) thin alpha_v dot  exp[-(alpha_v delta + beta)^2]. $ <metropolis_gaussian>
Significant features of this choice include that the drift term $-beta_v \/ alpha_v$ is mean–reverting
and the farther the initial state will be from the mean (zero), the lower the variance $1 \/ (2 alpha_v^2)$ will be.

This choice leads to an acceptance rate of the updates of $>=99.5%$.

This algorithm is very parallelizable, so we implemented it with `CUDA`. At each update, we randomly extract the parity of the firsts sites to update. So we update all sites with that parity and later the sites with the opposite one, in order to avoid data–race conditions.

== Cluster update
<cluster>
This algorithm is a generalization of the Wolff algorithm, not based on any symmetry of the system.

Instead of modifying one site per step, we build a cluster of adjacent sites, then change every site in it of the same quantity $delta$.
@cluster_building and @innovation_sampling give an outline of the algorithm, then all probabilities and parameters are
made explicit and discussed in detail in @prob_tuning.

=== Cluster building <cluster_building> \
The basic idea is to start from a randomly selected site $i^star$, then append neighbors to the cluster with probability
$P_"add" (Delta x_"neigh")$, which only depends on the kinetic energy contribution $Delta x_"neigh" = x_"neigh"-x_(i^star)$, as we show in the pseudo-code snippet below:
#pseudocode-list(hooks: .5em, indentation: 0.5em, line-gap: 0.5em)[
  + $#raw("cluster[0]") = i^star$ first cluster's site
  + $l = 1$ cluster's length; #h(1em) $k = 0$ counter
  + *while* $k < l$:
    + *for* $i$ first neighbor of `cluster[k]`
      + *if* $i in.not$ `cluster`
        + append $i$ with probability $P_"add" (x_i - x_#raw("cluster[k]"))$
        + $l <- l + 1$
    + $k <- k + 1$
]

=== Innovation sampling <innovation_sampling> \
Given the cluster $cal(C) = (m,...,M)$, of size $N_cal(C)$, we extract an innovation $delta$ from a probability density function
$g(delta thin | thin v, cal(C))$, so that the trial state will be
$ w = (x_0,..., x_m + delta,..., x_M + delta,..., x_(N-1)). $
We remark that the entire kinetic energy change from initial to final state are due $x_m$ and $x_M$ updates, and so that we can write the probability of building the cluster $cal(C)$ starting from $v$ as
$ P_"build" (cal(C) thin | thin v) &= P_"in" (cal(C),v) dot \
  & quad dot [1 - P_"add" (thin |x_m - x_(m-1)|thin)] dot \
  & quad dot [1- P_"add" (thin |x_M - x_(M+1)|)],  $
where $P_"in"$ only depends on the relative distances inside the cluster and it's invariant under a global shift of the cluster (and so $P_"in" (cal(C),v)=P_"in" (cal(C),w)$).
  
=== Probability tuning <prob_tuning> \
As before, we aim to maximize the acceptance probability
$ P_"acc" = min[1,
  (g(-delta thin | thin w,cal(C)) dot P_"build" (cal(C) thin | thin w) dot P(w))/
  (g(delta thin | thin v,cal(C)) dot P_"build" (cal(C) thin | thin v) dot P(v)) 
] $
by choosing the appropriate $g(delta thin | thin v, cal(C))$ and $P_"add" (Delta x_"neigh")$.
One can verify that an acceptable choice for these is:
$ P_"add" (Delta x_"neigh") = max{1 - gamma exp[+(Delta x_"neigh"^2)/(2 a)], 0}, $ <P_add>
for a suitable choice of $gamma$. Moreover, for the $g$ function, we can choose
$ cases(
    display(g(delta thin | thin v , cal(C)) = sqrt(2/pi) alpha_v exp[- (alpha_v delta + beta_v)^2] )",",
    display(alpha_v = sqrt(a/2  N_cal(C)) dot sqrt(1 + 3 g thin hat(x^2)))",",
    display(beta_v = a/2 dot N_cal(C)/alpha_v dot [hat(x) + g thin hat(x^3)])","
  ) $
where:
- the parameter $gamma$ is free to vary in the range $(0,1)$,
  and controls the average size of clusters, in addition to setting a maximum $Delta x_"neigh"$ that can be accepted;
- $hat(x), hat(x^2), hat(x^3)$ are meant as averages within the cluster.
In comparison to the single site update, we remark that now the drift term in the innovation pdf is independent of the kinetic energy variation, as that has been taken into account by $P_"add"$.

In our case, we chose $gamma$ in order to have the average cluster size of about $sqrt(N)$, with $N$ the total number of sites in the simulation. We can reach this goal observing that $Delta x$ is approximately distributed as
$
  prop e^(- Delta x^2\/2 a),
$
so, roughly, $Delta x_"typ" tilde sqrt(2 a)$; it follows than that we approximately want
$
  N dot (1-gamma / e) tilde sqrt(N),
$
that can be used for an estimate of $gamma$.

We implemented this algorithm both with single cluster update, completely running on CPU and written in plain `C++`, and multi–cluster update with `CUDA`, GPU parallelized.

In the latter case, we first building the clusters as in the single cluster algorithm.
In order to do that, on each site, a thread compute if the next site is in the same cluster of its.
Then, in order to avoid data–race conditions, in the graph that has the cluster as vertex and the interaction as edges, we color this graph and we update one color at a time.

Our parameters leads to an update acceptance ratio of $5%—30%$ for the multi-cluster update and $60%—90%$ for the single-cluster update.

All GPU simulations run on a `NVIDIA GTX 1650`.

As far as the random number generator is concerned, we use PCG32 @oneill:pcg2014, which is high-quality and high-speed.

= Data analysis
<data_anal>
Having algorithms that sample the path distribution, we now focus on how to find energy gaps and their uncertainties.
Our estimate of energy gaps fully relies on solving the following generalized eigenvalue problem (GEVP):
$ C(t + tau) v = lambda C(t) v, $ <gevp>
where $C(t)$ is the connected correlator matrix at lag $t$:
$ C_(i j) (t) = angle.l O_i (t) O_j (0) angle.r - angle.l O_i angle.r angle.l O_j angle.r. $ <def_correlator>

We only make use of observables that depend on the position operator, since the vectors $hat(x)^n ket(0)$ span all the Hilbert space of the states; given a state $s$ of the path and an operator $O(x)$,
our sample is:
$ overline(O)_s = 1/N sum_(i=0)^(N-1) O(x_(s,i)) , $
whereas our estimator for it's mean value $angle.l O angle.r$, given $N_s$ the (independent) samples number, will be:
$ hat(O) = 1/N_s sum_(n = 1) ^(N_s) #h(0.5em) overline(O)_n"." $

In particular, for each update, we computed the following observables:
- the first four powers of $x$, averaged along the path: $O_i = x^i; " " i=1,...,4$;
- the raw correlator between those powers $O_i (t) O_j (0)$.
In @stat_corr and @stat_gap we briefly describe how we compute statistical uncertainties, in @find_gaps how we use the GEVP @gevp to find the first four energy gaps, and finally in @pars_tuning we explain how to reach an arbitrary target precision.

== Statistical errors on correlators
<stat_corr>
Firstly, we remark that the correlators matrix $C_(i j) (t)$ is a secondary observable, for it depends on the primary observables $angle.l O_i angle.r$, $angle.l O_j angle.r$ and $angle.l O_i (t) O_j (0) angle.r$.
In order to compute the appropriate uncertainties, we rely on standard first order error propagation, using the following procedure:
+ starting from raw samples of primary observables, we block data to eliminate correlations due to the Markov Chain. Note that the number of samples reduces to $N_s^"eff"$;
+ we compute sample covariance between the primary observables;
+ we use the first order error propagation formula (using Einstein notation):
  $ "cov"[f_i,f_j] = (diff f_i)/(diff x_l) dot (diff f_j)/(diff x_m) dot "cov"[x_l,x_m] . $ <error_propagation>

== Finding energy gaps
<find_gaps>
With regard to @def_correlator, in the low temperature limit #footnote([$beta -> +oo$ quantities will be indicated with a tilde.]), the correlator takes the form:
$ tilde(C)_(i j) (t) = sum_(n=1)^(+oo) braket(0,O_i ,n) braket(n, O_j , 0) e^(-t(E_n - E_0)). $
Then in the large $t$ limit (see below for clarification), we neglect contribution over the fourth gap, and call $tilde(C)' _(i j) (t)$ the truncated correlator.

The solution to @gevp, with the $tilde(C)'$ instead of $tilde(C)$, is _exactly_:
$ tilde(lambda)'_n (tau) = exp[-tau(E_n - E_0)] , $ <eigenval2gap>
which can be inverted to find gaps.
These approximations introduce the following systematic errors#footnote([As discussed below, for the Hellmann–Feynman theorem, the error of the eigenvalue is proportional to the error of the matrices entries, so we will interchangeably use these notions.]):
- finite $beta$ introduces an error on each entry of the correlators' matrix:
  $ C_(i j) (t) approx tilde(C)_(i j) + O(e^(-beta (E_1 - E_0))); $ <beta_sys>
- finite $t$ introduces an error on the correlators' matrix entries:
  $ tilde(C)_(i j)(t) approx tilde(C)'_(i j) (t)[1 + O(e^(-t(E_p - E_(p-2))))], $ <corr_sys>
  where we call $E_p$ is the first not sampled energy level that has the same parity#footnote([The symmetry $x->-x$ induces a selection rule that make negligible contributions to the correlators of negative parity.]) as $n$;
- due to the presence of further states in the sampled $C$, the running of $lambda_n$ with $tau$ is not exactly exponential. This induces a correction in the computation of the eigenvalues whose dominant component scales with $ e^(-tau dot (E_p - E_n)). $ <lambda_sys>

== Statistical uncertainties on energy gaps
<stat_gap>
In order to estimate statistical uncertainties on the GEVP solutions, we use Eq. @error_propagation, computing derivatives via the Hellmann–Feynman theorem.
Let $lambda$, $v$ solve Eq. @gevp, and let $C_(i j k) = C_(i j)(t + k tau)$ with $k = 0,1$.
Then the theorem states that:
$ (diff lambda)/(diff C_(i j k)) = braket(v, diff/(diff C_(i j k)) [C_(i j 0) - lambda C_(i j 1)],v)
  / braket(v, C_(i j 1) ,v). $

== Parameter tuning
<pars_tuning>
We set $eta$ and $eta_"stat"$ the target relative errors respectively due to systematic and statistical errors.
We are particularly interested in assuring them to be reached on our fourth gap estimate, which presents the smallest signal and is the most sensitive to systematic errors by means of Eqq. @lambda_sys and @eigenval2gap.
Firstly, we ask that both errors in Eqq. @lambda_sys and @corr_sys are less than $eta$ and that the finite $beta$ in Eq. @beta_sys is negligible ($lt.tilde eta\/10$):
$ cases(
  display(tau ~ t gt.tilde (ln(eta))/(E_4 - E_6))",",
  display(beta gt.tilde (ln(eta\/10))/(E_0-E_1))"."
) $
Secondly, taking into account that statistical errors on $lambda$ are proportional to the ones on $C$, we ask the fourth gap signal to be greater than the noise, getting:
$ t lt.tilde ln(eta_"stat")/(E_1 - E_4). $
An important remark is that we're asking:
$ eta_"stat" lt.tilde e^(-t(E_4 - E_1)) < e^(-t(E_6-E_4)) lt.tilde eta , $
hence the statistical error will be negligible with respect to the systematic one.

== Discretization effects
The discreteness of the path must be taken into account as well.
For each value of the coupling parameter, we perform multiple simulations varying the time step parameter $a$, and use a quadratic fit on the energy gaps to estimate their value in the continuum limit as shown in . For this fit, only statistical uncertainties are taken into account.
#figure(
  image("pics/scaling_metropolis_000784.svg"),
  caption: [Quadratic fit of the energy gaps for a simulation performed with single site update algorithm and coupling parameter $g = 7.84$]
)

= Results
<results>
We finally present the result of our simulations.
We used a relative error of $30%$ for the multi-cluster update algorithm, and $1%$ for the single site and single cluster update.
As a comparison, we computed the gaps with a first order Perturbative–Variational approach, well explained in @paffuti.

In conclusion, we see a good agreement between simulations and the Perturbative–Variational estimate of the energy gaps.

We are confident that great performance improvements can be obtained using a more suitable GPU (or higher dimensionality simulations) for the multi–cluster algorithm.
However, single site update seems to stay the best option for its cleanness and overall performance.

= Acknowledgements
#link(
  "https://github.com/mbar02/nummet-public/raw/refs/heads/master/report/Module%201/pics/2mhh3d0hx1gb1.png",
  text(fill: blue, size: 0.8em, underline(raw("https://github.com/mbar02/nummet-public/raw/refs/heads/master/report/Module%201/pics/2mhh3d0hx1gb1.png")))
)
#bibliography("refs.bib", style: "springer-basic", full: true)

#place(scope: "parent", float: true, bottom, [
= Figures
  #figure(
    grid(columns: 2,
      image("pics/gaps_coupling_metropolis.svg", height: 23em),
      image("pics/gaps_coupling_wolff.svg", height: 23em),
      grid.cell(colspan:2, image("pics/gaps_coupling_multicluster.svg", height: 23em, )),

    ),
    caption: [Energy gaps varying the coupling parameter $g$, computed with all three different algorithms: single site (top left), single cluster (top right), multi-cluster (bottom). The dotted lines are the gaps computed with the Perturbative-Variational method.]
  )
]
)