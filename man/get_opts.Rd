\name{get_opts}
\alias{get_opts}
\title{
  Format parameters for the prior distribution of mintMR
}
\description{
  This function creates a list of parameters for the prior distribution for mintMR. It also controls parameters for the iteration process.
}
\usage{
get_opts(L, a_gamma = NULL, b_gamma = NULL, a_alpha = NULL, b_alpha = NULL, 
         a_beta = NULL, b_beta = NULL, a = NULL, b = NULL, maxIter = NULL, 
         thin = NULL, burnin = NULL)
}
\arguments{
  \item{L}{
    The number of genes to be jointly analyzed in the model.
  }
  \item{a_gamma, b_gamma}{
    Parameters for the prior distribution of the true IV-to-Exposure effect. They represent the shape and scale parameters, respectively.
  }
  \item{a_alpha, b_alpha}{
    Parameters for the prior distribution for the direct effect.
  }
  \item{a_beta, b_beta}{
    Parameters for the prior distribution for causal effects.
  }
  \item{a, b}{
    Parameters for the prior distribution for the latent status.
  }
  \item{maxIter}{
    The maximum number of iterations for the Gibbs sampling.
  }
  \item{thin}{
    The frequency of sample retention in Gibbs sampling.
  }
  \item{burnin}{
    The number of burn-in iterations.
  }
}
\value{
  \item{a_gamma, b_gamma, a_alpha, b_alpha, a_beta, b_beta, a, b}{
    Parameters for the prior distributions.
  }
  \item{maxIter, thin, burnin}{
    Control parameters for the Gibbs sampling process.
  }
}
\examples{
# Example usage of get_opts for a Bayesian hierarchical model]
L = 100
opts <- get_opts(L = L, 
                 a_gamma = rep(1,L), b_gamma = rep(0,L), 
                 a_alpha = rep(0,L), b_alpha = rep(0,L), 
                 a_beta = rep(1,L), b_beta = rep(0.1,L), 
                 a = 0.1, b = 0.1, 
                 maxIter = 4000, thin = 10, burnin = 1000)
}

\keyword{bayesian}
\keyword{genetics}
