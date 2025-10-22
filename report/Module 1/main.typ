//==============================================================================
// ifacconf.typ 2023-11-17 Alexander Von Moll
// Template for IFAC meeting papers
//
// Adapted from ifacconf.tex by Juan a. de la Puente
//==============================================================================

#import "@preview/abiding-ifacconf:0.2.0": *
#import "@preview/physica:0.9.5": *

#show: ifacconf-rules
#show: ifacconf.with(
  title: [Introduction to MCMCs \ Module 1: Critical properties of the Ising model],
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
The aim of this report is to verify some critical properties of the Euclidean $D=2$ Ising model through Monte Carlo simulations.
We focus on critical temperatures and critical exponents $gamma, nu,$ and $z$, which  determine all the others – via scaling relations.
The simulations are performed on square, triangular, and hexagonal lattices, with toroidal boundary conditions.
Data from Wolff and Metropolis algorithms are compared.
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
#let tableFromCSV(filename, column-gutter: auto, cell-inset: auto, ..args) = {
  let csvfile = csv("example.csv")
  let csvfile-data = csvfile.slice(1).map( row=>{
    row.map( cell=>{
      eval(cell, mode: "markup")
    })
  })

  set table.cell(inset: cell-inset, align: horizon)
  
  table(
      columns: csvfile.first().len(),
      column-gutter: column-gutter,
      table.header( ..csvfile.first().map( title => { eval(title, mode: "markup") } ) ),
      ..csvfile-data.flatten(),
      ..args
  )
}

= Introduction

The Ising model is the simplest statistical model for a ferromagnetic system.
It consists of a lattice of spins $sigma_i$, which can take only value $+1$ and $-1$, and the Hamiltonian has form
$
H = - J sum_(angle.l i j angle.r) sigma_i sigma_j -sum_i h_i sigma_i,
$
where by $angle.l i j angle.r$ we mean that the sum only runs on adjacent sites, and $h_i$ is an external background magnetic field.
The energy manifestly has a $ZZ_2$ symmetry.

The dynamics of said system, in two or more dimensions, spontaneously breaks this symmetry at non-vanishing temperature, leading to a phase transition.
In two dimensions, the critical behavior of the Ising model was first fully described by L.~Onsager.

In this report, we verify the theoretical results through numerical simulations.
The discussion is organized as follows:
- in @simulations we briefly summarize the procedures, choices, and conventions we used for the simulations;
- in @data_anal we present the statistical and fitting methods that have been used to analyze the numerical simulations;
- in @results we show and briefly comment on the obtained results.


= Methods of Investigation
<simulations>

== Units
Throughout the document, we set the Boltzmann constant $k_"B"=1$ and we work in energy units of $J$, therefore $J=1$. 

== Simulation algorithms
In order to investigate critical properties of the model, we rely on Monte Carlo simulation of the observables, with the underlying distribution being the usual Boltzmann distribution:
$ P(sigma_1, ... sigma_n) = 1/Z dot exp[-beta H(sigma_1,..., sigma_n)] . $
The only observables that we need for our purposes are the average magnetization per unit volume $m$, its absolute value $abs(m)$ and its square power $m^2$.
Two different Markov Chains are built to sample their distribution:

=== The Metropolis MC
with deterministic choice of sites to be updated; for the associated Markov Chain to be aperiodic, it is necessary that at each step an extraction is made in each spin of the lattice.
This procedure is highly parallelizable; therefore, we implemented it via the GPU (Apple Silicon M1 Pro, in our case).
This program is written in `C++` and `Metal`#footnote([For an introduction on how to use Metal for scientific computing on Apple Silicon processors, see @Gebraad_2023.\ The repo attached to that paper does not free the allocated Metal objects, and this for long simulations can lead to large amounts of allocated RAM. Because of the way Metal handles memory, this must be done via `object->release()` and not via `delete object`. Moreover, this resource does not describe asynchronous communication between CPU and GPU, which could be a relevant feature in a performance sensitivity context.]).

To avoid data races, we first update sites with even abscissa and even ordinate are updated, then those with even abscissa and odd ordinate, etc.... This works straightforwardly with the three geometries.

The associated Markov Chain is manifestly irreducible and aperiodic.
However, there is a subtle point about the balance condition: in fact, during each of the four moments in a step, the Markov Chain associated to the transformation is not irreducible (e.g. during the first moment, it does not update spins with odd abscissa and ordinate).
Nevertheless, the balance condition is satisfied, as each of the four sub-chains samples the conditioned probability.

#boring_proof([
Let us consider a system composed of two subsystems $A$ and $B$. Let $cal(H)_A$ and $cal(H)_B$ the free vector spaces generated by the states of $A$ and $B$, respectively, and let $pi_(a,b)$ (with $a in A, b in B$) the p.d.f.~to be sampled. The analogous of the sub-chains that update sites of given parity of abscissa and ordinate, in this case, is a pair of irreducible (on the subsystem) and aperiodic Markov Chain $cal(M)_1, cal(M)_2$ that act only on the first or on the second subsystem, in the same order, but that have as asymptotic p.d.f.~the conditioned probability $pi_(i|j)$ and $pi_(j|i)$.
Let $W_i$ be the stochastic matrix associated with $cal(M_i)$. We can state the balance condition as
$
  forall b_0 in B, quad W_1 sum_a pi_(a|b_0) ket(a) otimes ket(b_0) = sum_a pi_(a|b_0) ket(a) otimes ket(b_0),quad\
  forall a_0 in A, quad W_2 sum_b pi_(b|a_0) ket(a_0) otimes ket(b) = sum_b pi_(b|a_0) ket(a_0) otimes ket(b).quad
$
Using Bayes theorem, it is possible to show that
$
ket(pi) &=sum_(a,b) pi_(a,b) ket(a)otimes ket(b) \
        &=sum_(b_0) pi_b_0 sum_(a) pi_(a|b_0) ket(a)otimes ket(b_0) \
        &=sum_(a_0) pi_a_0 sum_(b) pi_(b|a_0) ket(a_0)otimes ket(b) \
$
and so is an eigenvector with unitary eigenvalue of $W_1$ and $W_2$, so it is also an eigenvector of unitary eigenvalue of $W=W_1 dot W_2$, and that concludes the proof.

The generalization in the case of four sub-systems (that is our case) is trivial.
])

=== Wolff's MC
a single-cluster algorithm that is hard to parallelize but much more efficient than Metropolis, especially close to the critical point. This program is written in plain `C`.

As far as the random number generator is concerned, we use PCG32 @oneill:pcg2014, which is high-quality and high-speed.

We use AoS memory management. This proves to be not only better for code readability, but even more efficient because it allows us to identify neighboring sites via pointers#footnote([Attention must be paid to some subtleties regarding how these addresses are created. For example, code running on the GPU sees different virtual addresses than on the CPU.]).

Both simulation algorithms accept as arguments a `u.p.s.`--updates per sample parameter that allows only one data item to be saved every `u.p.s.`~steps of the simulation.

All codes are available at our repository on GitHub @RepoGH.

== Lattices
Simulations are carried out on two-dimensional lattices with periodic, untwisted boundary conditions.

#block([
#figure(
  image("pics/sqr_lattice.svg", height: 3cm),
  caption: [Lattice with square geometry.],
  placement: auto
) <sqrLattice>
#figure(
  image("pics/tri_lattice.svg", height: 3cm*calc.sqrt(3)/2),
  caption: [Lattice with triangular geometry.],
  placement: auto
) <triLattice>
#figure(
  image("pics/hex_lattice.svg", height: 3cm*calc.sqrt(3)/2),
  caption: [Lattice with hexagonal geometry. A unit cell is shadowed.],
  placement: auto
) <hexLattice>
])

In order both to compare critical temperatures with expected temperatures and to verify the _universality_ of critical exponents, we adopt three lattice families:
- _square geometry_: with $(i;j)$ Cartesian coordinates, lattice of $L times L$ points;
- _triangular geometry_: with $(i;j)$ non-orthogonal Cartesian coordinates, and a lattice of $L times L$ spins;
- _hexagonal geometry_: with coordinates $(i;j;k)$ and a lattice $L times L times 2$; the compact coordinate $k$ discides the spin within the unit cell.
A visual representation of the three is shown in @sqrLattice, @triLattice and @hexLattice.\
In both cases, we save spins in RAM in row-major order:\
#h(1fr)`lattice[i,j] := lattice[i*L+j]`#h(1fr)\
for square and triangular lattices and\
#h(1fr)`lattice[i,j,k] := lattice[i*2L+2j+k]`#h(1fr)\
for the hexagonal lattice.

Each spin in the lattice is represented with a `struct`\
#align(center,```
struct cell {
 int64_t      value;
 struct cell* neighbors[N_NEIGH];
};
```)
containing the spin value and neighbors. Important performance increase is obtained using pointers to neighbors instead of indices with the Wolff algorithm#footnote([By using indices instead of pointers, the run time worsens by a factor of 1.5.]), but the opposite in the case of Metropolis.

To avoid unnecessary “if-checks” in functions, we use macros and compile three versions of each program, one per lattice geometry.

= Data analysis
<data_anal>
This section attempts to analyze the subtleties we found mainly in the treatment of uncertainties and finite size effects.

In the Ising model, the order parameter is the expectation value of the magnetization
$
expval(m)=angle.l sigma_i angle.r
$ ($m$ does not depend on $i$ because of translation invariance).

The S.S.B.~of the critical (infinite) system manifests itself in the finite system behavior with the emergence of two marked peaks into the probability density function of the average magnetization, at low temperatures. At higher $L$, the two peaks become sharper, as shown in figure @ssb.

== Critical temperature and exponents
<critical_temperature_and_exponents>
We now introduce the reduced magnetic susceptibility, which is the core of our investigation, defined as follows:
$
chi = beta V [angle.l m^2 angle.r - angle.l |m| angle.r ^2 ] = beta V sigma^2_(|m|),
$
where $beta = 1\/T$ is the inverse temperature.

In the thermodynamic limit, $chi$ has a point of non-analyticity at the critical temperature, which allows to define the critical exponent#footnote[_A priori_, the limits for $beta->beta_"cr"^pm$ could lead to different $gamma$s. This is not the case for the Ising model.] $gamma$:
$ chi tilde |beta - beta_"cr"|^(-gamma). $

In a finite lattice, clearly such a point of non-analyticity does not exist. Therefore, finite size effects must be taken into account in order to calculate the critical temperature.
Critical exponents $gamma$ and $nu$ can be found by looking at the scaling laws related to the magnetic susceptibility; in particular, calling $chi^((L)) (beta)$ the susceptibility associated to a given lattice of size $L$, must follow:
$ chi^((L)) (beta) = L^(gamma/nu) dot f(L^(1/nu) dot (beta - beta_"cr")) dot [1+O(1/L)] $
where $f$ is called _universal function_.
Furthermore, in our case the universal function is unimodal, therefore peak coordinates $(beta_"max"^((L)); chi_"max"^((L)))$ of susceptibility satisfy the scaling relations:
$ cases(
  beta^((L))_"max" simeq beta_"cr" + b dot L^(-1/nu),
  chi^((L))_"max" simeq c_0 + c_1 dot L^(gamma/nu)
) $ <chi_scaling>
Then we estimate the values of critical temperature and exponents by the following procedure:
+ chosen the lattice side $L$, sample the statistical distribution at various $beta$;
+ for each sample, we compute the magnetic susceptibility, and we identify its peak coordinates via a quadratic fit, as shown in @chi_peak;
+ we repeat 1.~and 2.~varying $L$;
+ we extract critical temperature and exponents via an exponential fit, using @chi_scaling, as shown in @scaling_metropolis and @scaling_wolff.
We consider the following systematic errors:
- the pseudo-random number generator generates #box("32-bit") integers. We expect this to shift the simulation temperatures by about $2^(-32)$, therefore is completely negligible.
- To find the maximum of $chi_L$, we sample it around its maximum and perform a fit with a parabolic model; systematic errors arise from neglected higher-power contributions. 

#figure(
  image("final-plots/chiMax_metropolis_hex_80.svg", width: 100%),
  caption: [Example of quadratic fit for the maximum of the susceptibility.\ This was performed at $L=80$ in an hexagonal lattice.]
)
<chi_peak>
  
In order to neglect the latter, we tune simulation parameters so that the statistical error is about 10 times larger than the systematic one (and so the uncertainties can be estimated via a standard statistical data analysis), proceeding as follows:
+ we perform rough preliminary measurements to estimate $beta_"max"^((L)), chi_"max"^((L))$, $m$ and $m^2$ sample variances, and the integrated correlation time, as shown in @pre_sqr, @pre_tri, and @pre_hex;
+ we expect that by expanding $chi_L(beta)$ around the maximum,
  $
  chi^((L))(beta) = chi_"max"^((L)) &- 1/2 ( (beta - beta_"max"^((L))) / beta_0^((L)) )^2 \
  &+ 1/6 chi_3^((L)) ( (beta - beta_"max"^((L)) ) / beta_0^((L)) )^3 + ..., #h(2em)
  $
  where $chi_3^((L))$ is a numerical factor in the order of unity;
+ we make new simulations in which we tuned the parameters in order to have statistical uncertainty about ten times larger than the cubic term, but ten times smaller than the quadratic one, to have a sufficient signal-to-noise ratio. In order to have so, we estimated the correct number of samples from preliminary sample variance, the `u.p.s.` from the integrated correlation time, and temperature range from $beta_"max"^((L)), chi_"max"^((L))$:
  $ cases(
  display(sigma^2 simeq 9/(2 dot 10^4) (chi_"max"^((L)))^(-1)),
  display(Delta beta^((L)) = sqrt(3/(5sqrt(2))) thick beta_0^((L)) (chi_"max"^((L)))^(1\/4)\
  #h(3em)  approx  0.65 thick beta_0^((L)) (chi_"max"^((L)))^(1\/4)))
  $
  In our case, we sampled 10–15 points in that interval, as shown in @chi_peak.


== Statistical uncertainties of primary observables
<incertezza_primarie>
With regard to the primary observables, we make two important remarks.
Firstly, two data generated a few steps apart are highly correlated, then we need to be cautious when treating statistical uncertainties.
Secondly, it is desirable to save as little data as possible, for it takes up time and memory storage to do so.
Therefore, we must save data once every `u.p.s.` steps of the simulation (an integer parameter), so as to solve both of these problems.
In order to find the right value for this parameter, we need to cope with two quantities, the exponential autocorrelation time $tau_"exp"$ and the integrated exponential time $tau_"int"$.

=== Estimate of $tau_"exp"$
<tempo_correlazione>
In the Markov Chains Monte Carlo formalism,
$
tau_"exp" = -ln(Lambda),
$
where $Lambda$ is the largest modulus eigenvalue other than 1 in the spectre of the stochastic matrix $W$.
We stress that the exponential autocorrelation time is intrinsically related to the Markov chain, as for any observable $O$, letting $C_O (k)$ be the autocorrelation relative to $O$ at lag $k$, it is true that, for almost every initial vector in the free vector space of states:
$ C_O (k) quad attach(tilde, t:k->oo) quad exp[-k/tau_"exp"] $
Furthermore, we point out that the autocorrelation time follows a scaling law, close to critical temperature. That is known as critical slowing down:
$ tau_"exp" tilde L^z  $
that defines a new critical exponent $z$.
In order to characterize it, after finding the critical temperature as explained in @critical_temperature_and_exponents, we perform simulations at $beta_"cr"$ for different $L$s and we extract $tau_"exp"$ via an exponential fit, as shown in @autocorr_fit; finally, we estimate $z$ by means of another exponential fit, as shown in @scaling_tau_metropolis and @scaling_tau_wolff.

#figure(
  image("pics/autocorr-example.svg"),
  caption: [Exponential decay of autocorrelation; orange curve is the best fit exponential.])
  <autocorr_fit>

=== Estimate of $tau_"int"$
This second autocorrelation time is deeply related to the variance of the sample average, and it depends on the particular observable taken into account.
In fact, given the observable $O$ with $sigma^2_O$ variance, let $overline(O) = 1/N sum_(i = 1)^N O_i$ be the sample mean.
It can be shown that the variance of this random variable is:
$ sigma^2_(overline(O)) = sigma^2_O /N^2 dot sum_(i=1)^N sum_(j-i) C_O (|i-j|) simeq^((*)) sigma^2_O /N (1 + 2 tau_"int"^O), $ <tau_int>
where we defined:
$ tau_"int"^O = sum_(k=1)^(+oo) C_O (k). $
We point out that in @tau_int.($*$) we extend the series up to infinity as following contribution are exponentially damped by $tau_"exp"$.
For us to be able to identify the integrated autocorrelation time then, we follow a _blocking_ procedure: starting from the original sample $O_1;...; O_N$ we build a reduced one where
$ O_i^k = 1/k sum_(j = k dot i + 1 ) ^(k dot (i+1)) O_i $
and compute the sample variance.
As we increase $k$, the measures in the reduced sample get more and more uncorrelated, until the sample variance of the mean saturates to $sigma^2_(overline(O))$, as shown in @blocking.
We implemented an algorithm that estimates the smallest size $k^star$ for which the saturation occurs.

Finally, we compute the sample variance $sigma^2_F$ and use @tau_int to find $tau_"int"^O$.
At critical temperature, the integrated autocorrelation time of the observable $abs(m)$ follows a scaling law, as well:
$ tau_"int" ^abs(m) tilde L^(z'). $
We compute this exponent via an exponential fit, shown in @scaling_tau_metropolis and @scaling_tau_wolff.

#figure(
  image("pics/blocking.svg",width:90%),
  caption:[Value of the sample variance of the blocked sample, varying the block size, for the average magnetization and energy.
           Diamond marks the $k^star$.]
)
<blocking>

== Statistical uncertainties of secondary observables
For the purposes of this report, we only need reliable estimates of the uncertainties for the magnetic susceptibility,
as it serves to fit the critical inverse temperature $beta_"cr"$ and the critical exponents $gamma$ and $nu$.
Being a secondary observables (it depends on both $m^2$ and $|m|$), we use a Jackknife method: using the blocked data, we start from $N$ measures
of our primary observables and generate $N$ leave-one-out mock samples; one can prove that:
$ sigma_(overline(O)) simeq N dot (overline(O^2_J) - overline(O_J)^2) $
with $overline(O)_J$ being the mean of the mock samples.

The two procedures (blocking and Jackknife) give robust statistical meaning to the uncertainties of the magnetic susceptibility, and guarantee that
it is sufficient to launch longer simulations in order to get an arbitrary precision.

= Results
<results>
We finally present the results obtained with different geometries and algorithms.
All experimental measurements are available at @tabellariassuntiva.

Thanks to the attention paid in the choice of the sampling intervals of $beta$  and to the number of measurements, most of the quadratic fits
performed to find the peak of $chi$ show a $chi^2 \/"d.o.f"$ compatible to $1$ and all of them compatible within
$2 sigma$.
We recall that the fit for $tau_"exp"$ was done for operational needs (less data, uncorrelated measures), and the error associated with that result has no statistical meaning.

= Conclusion
The most interesting conclusion we can extract from this work is the comparison between the Metropolis and Wolff algorithms; both of them are able to simulate the Ising model, giving results that are well in agreement with the theory.
Nevertheless, as far as efficiency is concerned, Metropolis shows a much heavier critical slowing down close to critical point, and in general greater correlation times, even though the steps are whole lattice updates
and `GPU`'s raw computing power is being used.

#colbreak()

= Acknowledgements
#align(center, block(
  image("./pics/2mhh3d0hx1gb1.png", width: 60%)
))

#bibliography("refs.bib", style: "springer-basic")

#colbreak()

= Figures and tables

#place(scope: "parent", float: true, bottom + center, [
  #figure(
    image("plots-chi/chi_wolff_sqr.svg", width: 55%),
    caption: [Preliminary measurement of the reduced susceptibility of the square lattice.]
  )
  <pre_sqr>
  #figure(
    image("plots-chi/chi_wolff_tri.svg", width: 55%),
    caption: [Preliminary measurement of the reduced susceptibility of the triangular lattice.]
  )
  <pre_tri>
  #figure(
    image("plots-chi/chi_wolff_hex.svg", width: 55%),
    caption: [Preliminary measurement of the reduced susceptibility of the hexagonal lattice.]
  )
  <pre_hex>
])
#place(scope: "parent", float: true, auto, [
  #figure(
    grid(columns: 2,
      image("final-plots/chi_scaling_metropolis.svg", width: 90%),
      image("final-plots/beta_scaling_metropolis.svg", width: 90%)
    ),
    caption: [Scaling of $beta_"max"$ and $chi'_"max"$ with Metropolis algorithm.]
  )
  <scaling_metropolis>
])
#place(scope: "parent", float: true, auto, [
  #figure(
    grid(columns: 2,
      image("final-plots/chi_scaling_wolff.svg", width: 90%),
      image("final-plots/beta_scaling_wolff.svg", width: 90%)
    ),
    caption: [Scaling of $beta_"max"$ and $chi'_"max"$ with Wolff algorithm.]
  )
  <scaling_wolff>
])
#place(scope: "parent", float: true, auto, [
  #figure(
    grid(columns: 2,
      image("pics/binning_tri_30_261.svg", width: 90%),
      image("pics/binning_tri_30_282.svg", width: 90%),
      image("pics/binning_tri_300_261.svg", width: 90%),
      image("pics/binning_tri_300_282.svg", width: 90%),
    ),
    caption: [Histograms of the sampled probability of total magnetization. The four picture are related to a triangular lattice. The two at the top have been sampled with $L=30$, the two at the bottom with $L=300$. At left, $beta=0.261$ (above the critical temperature) and the right ones at $beta=0.282$.]
  )
  <ssb>
])
#place(scope: "parent", float: true, auto, [
  #figure(
    grid(columns: 2,
      image("pics/tau_int_metropolis_sqr.svg", width: 90%),
      image("pics/tau_exp_metropolis_sqr.svg", width: 90%),
      image("pics/tau_int_metropolis_tri.svg", width: 90%),
      image("pics/tau_exp_metropolis_tri.svg", width: 90%),
      image("pics/tau_int_metropolis_hex.svg", width: 90%),
      image("pics/tau_exp_metropolis_hex.svg", width: 90%),
    ),
    caption: [Scaling of the magnetization $tau_"int"$ (on the left column) and $tau_exp$ of the MCMC (on the right one) for square, triangular, and hexagonal lattices, respectively, with Metropolis algorithm.]
  )
  <scaling_tau_metropolis>
])
#place(scope: "parent", float: true, auto, [
  #figure(
    grid(columns: 2,
      image("pics/tau_int_wolff_sqr.svg", width: 90%),
      image("pics/tau_exp_wolff_sqr.svg", width: 90%),
      image("pics/tau_int_wolff_tri.svg", width: 90%),
      image("pics/tau_exp_wolff_tri.svg", width: 90%),
      image("pics/tau_int_wolff_hex.svg", width: 90%),
      image("pics/tau_exp_wolff_hex.svg", width: 90%),
    ),
    caption: [Scaling of the magnetization $tau_"int"$ (on the left column) and $tau_exp$ of the MCMC (on the right one) for square, triangular, and hexagonal lattices, respectively, with Wolff algorithm.]
  )
  <scaling_tau_wolff>
])
#place(scope: "parent", float: true, auto, [
  #figure(
    context tableFromCSV("example.csv", cell-inset: (x: 0.8cm)),
    caption: [Experimental results.],
    kind: "Table",
    supplement: "Tab."
    )
  <tabellariassuntiva>
])