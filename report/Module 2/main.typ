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
  title: [Markov chain Monte Carlo methods for path integral - Thermodynamics and energy gaps of the anharmonic oscillator],
  
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
The aim of this report is to study the low temperature behavior of an harmonic oscillator with $q^4$ perturbation,focusing in particular on its thermodynamic properties and first few energy gaps.
Whole-path, single cluster and multi-cluster update algorithms are presented and compared.
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
The system of interest of this report is a single quantum harmonic oscillator, with an anharmonic $q^4$ perturbation.
We focus on thermodynamical properties and energy gaps of such system in the so called thermal state, which is a mixture of Hamiltonian eigenstates so that the density matrix $rho$ corresponds to a Boltzmann distribution:

$ cases(
  H   = p^2 /(2 m) + m omega^2 /2 q^2 + g/4 q^4,
  rho = 1/Z(beta) dot exp[-beta H]
) $

The discussion is organized as follows:
- In @simulations we briefly summarize the procedures, choices and conventions used for the simulations;
- In @data_anal we present the statistical and fitting methods used to analyze the numerical simulations;
- In @results we show and comment on the obtained results


= Methods of investigation
<simulations>
== Units
Throughout the document, we set the Boltzmann constant $k_"B" = 1$.
Also, energy is measured in units of the harmonic oscillator gap, therefore $hbar omega = 1$.
Finally, length are expressed in units of harmonic oscillator characteristic length, hence $hbar/(m omega) = 1$

== Path integral and discretization
In order to investigate the wanted properties, we rely on Monte Carlo simulation of the observables.
As far as the underlying distributions, standard arguments imply that given the system's hamiltonian density $cal(H)$, the respective eucledian action $S_E$ and an observable $O$ depending on the position operator only, the following relations hold:
$ S_E [x(tau)] = integral_(x(0)=x(tau)) #h(-2.5em) cal(H) [x(tau)] dd(tau) $
$ &angle.l O angle.r = tr[O dot rho] = \
  & quad 1/Z(beta) dot integral_(x(0)=x(beta))  #h(-2.5em) cal(D)
  [x(tau)] O(x(tau)) thick exp[-S_E [x(tau)] ] $
which is the path integral representation of the expected value.
It is clear how the term $1/Z(beta) exp[- S_E]$ can be interpreted as a probability density function; we can discretize the integrals introducing a time step $a = beta/N$, getting:
$ P(x_0, ..., x_(N-1) = x_0) = 1/Z(beta) dot exp[-a sum_(i=0)^(N-1) cal(L) [x_i]] $
which is to be sampled for our investigation.
To this end, we build two different Markov chains, briefly descripted in @single_site and @cluster; each one will produce, at each iteration, a new path $(x_0,...,x_(N-1))$ which will be referred to as _state_ of the Markov chain.
We point out that both algorithms are of the Metropolis-Hastings type, and they are only different in how the trial state is chosen.

== Single site update
<single_site>
With this first Markov chain, the initial and trial state only differ by one point of the discretized path.
Given its index $i^star$, and the current state $v = (x_1,...,x_(i^star),...,x_(N-1))$, the trial state becomes
$ w = (x_0,...,x_(i^star) + delta,...,x_(N-1)) $ were the innovation $delta$ must be extracted form a probability density function
$g(delta thin | thin v, i^star)$.
Finally the trial state can be accepted with probability
$ P_("acc") = min(1,g(-delta thin | thin w, i^star)/g(delta thin | thin v, i^star) dot P(w)/P(v)) $
Calling $D^2$ the discrete laplacian, we observe:
$ #h(-1em) cases(
    display(P(w)/P(v) = exp[-(alpha_v delta - beta_v)^2  + beta_v^2 - a g x_(i^star) delta^3 - (a g)/4 delta^4]) ,
    display(alpha_v = sqrt(a/2) dot sqrt(2/a^2 +1 + 3 g x_(i^star)^2)) ,
    beta_v = 1/2 thin a thin alpha_v^(-1) [g x_(i^star)^3 + x_(i^star) - D^2 x_(i^star)]
  ) $
We point out that the two coefficients defined above actually depend on initial and final state,
but only indexed with the former for notation clarity.\
At this point, the better $g(delta thin | thin v, i^star) \/ g(-delta thin |thin w, i^star)$ resembles $P(v) \/ P(w)$, the closer the acceptance probability will be to one.
We also take into account that we need to sample $g(delta thin | thin v)$ for every simulation update, therefore a good choice of this distribution is
the gaussian in @metropolis_gaussian , which returns a decent acceptance probability, being easy to sample via Box-Müller.
$ g(delta thin | thin v , i^star) = sqrt(2 / pi) thin alpha_v dot  exp[-(alpha_v delta + beta)^2] $ <metropolis_gaussian>
Significant features of this choice include that the drift term $-beta_v \/ alpha_v$ is mean reverting
and the farther the initial state will be from the mean (zero), the lower the variance $1 \/ (2 alpha_v^2)$ will be.

We implemented this algorithm with `CUDA` GPU parallelizing, alternating update on even and odds sites.


== Cluster update
<cluster>
Instead of modifying one site per step, we build a cluster of adjacent sites, then change every site in it of the same quantity $delta$.
@cluster_building and @innovation_sampling give an outline of the algorithm, then all probabilities and parameters are
made explicit and discussed in detail in @prob_tuning. \
=== Cluster building <cluster_building> \
The basic idea is to start from a randomly selected site $i^star$, then append neighbors to the cluster with probability
$P_"add" (Delta x_"neigh")$, which only depends on the kinetic energy contribution $Delta x_"neigh" = x_"neigh"-x_(i^star)$, as we show in the pseudo-code snippet below:

#pseudocode-list[
  + $#raw("cluster[0]") = i^star$ first cluster's site
  + $l = 1$ cluster's length
  + $k = 0$ counter
  + *while* $k < l$:
    + *for* $i$ first neighbor of `cluster[k]`
      + *if* $i in.not$ `cluster`
        + append $i$ with probability $P_"add" (x_i - x_#raw("cluster[k]"))$
        + $l <- l + 1$
    + $k <- k + 1$
]


=== Innovation sampling <innovation_sampling> \
Given the cluster $cal(C) = (x_m,...,x_M)$, of size $N_cal(C)$, we extract an innovation $delta$ from a pdf
$g(delta thin | thin v, cal(C))$, so that the trial state will be
$ w = (x_0,..., x_m + delta,..., x_M + delta,..., x_(N-1)) $
We remark that only kinetic energies changing from initial to final state are $Delta x_(m-1)$ and $Delta x_(M+1)$, and that we can write the probability of building the cluster $cal(C)$ starting from $v$ as
$ & #h(-1.5em) P_"build" (cal(C) thin | thin v) = \
  & #h(-1.5em) quad P_"in" (cal(C)) dot (1 - P_"add" (Delta x_(m-1))) dot (1- P_"add" (Delta x_(M+1)))  $
where $P_"in"$ only depends on the relative distances inside the cluster and it's invariant under a global shift of the cluster.
  
=== Probability tuning <prob_tuning> \
As before, we aim to maximize the acceptance probability
$ & P_"acc" =\  & quad min[1,
  (g(-delta thin | thin w,cal(C) + delta) dot P_"build" (cal(C) + delta thin | thin w) dot P(w))/
  (g(delta thin | thin v,cal(C)) dot P_"build" (cal(C) thin | thin v) dot P(v))
] $
by choosing the appropriate $g(delta thin | thin v, cal(C))$ and $P_"add" (Delta x_"neigh")$.
One can verify that an acceptable choice for these is:
$ P_"add" (Delta x_"neigh") = 1 - gamma exp[(Delta x_"neigh"^2)/(2 a)] $ <P_add>
$ cases(
    display(g(delta thin | thin v , cal(C)) = sqrt(2/pi) alpha_v exp[- (alpha_v delta + beta_v)^2] ),
    display(alpha_v = sqrt(a/2  N_cal(C)) dot sqrt(1 + 3 g thin hat(x^2))),
    display(beta_v = a/2 dot N_cal(C)/alpha_v dot [hat(x) + g thin hat(x^3)])
  ) $
where:
- $P_"add"$ is one (zero) if the expression in eq. @P_add is greater than one (less than zero);
- consequently, the parameter $gamma$ is free to vary in the range $(0,1)$,
  and controls the average size of clusters, in addition to setting a maximum $Delta x_"neigh"$ that can be accepted;
- $hat(x), hat(x^2), hat(x^3)$ are meant as averages within the cluster.
In comparison to the single site update, we remark that now the drift term in the innovation pdf is independent of the kinetic energy variation, as that has been taken into account by $P_"add"$.
\ [STIMA DI COME GAMMA CONTROLLA CLUSTER SIZE] \

We implemented this algorithm both with single cluster update, completely running on CPU, and multi-cluster update with `CUDA` GPU parallelizing, first building the clusters, randomly choosing the "cluster zero" and finally updating odds only. 

= Data analysis
<data_anal>
Having algorithms that sample the path distribution, we now focus on how to find energy gaps and their uncertainties.
Our estimate of energy gaps fully relies on solving the following generalized eigenvalue problem (GEVP):
$ C(t + tau) v = lambda C(t) v $ <gevp>
where $C(t)$ is the connected correlator matrix at lag $t$:
$ C_(i,j) (t) = angle.l O_i (t) O_j (0) angle.r - angle.l O_i angle.r angle.l O_j angle.r $
We recall that, in the large $beta$ limit (low temperature), the connected correlator

We only make use of observables that depend on the position operator; given a state $s$ of the path and an operator $O(x)$,
our estimator for its mean value $angle.l O angle.r$ is:
$ overline(O)_s = 1/N sum_(i=0)^(N-1) O(x_(s,i)) $







= Results
<results>

= Conclusion
<conclusion>