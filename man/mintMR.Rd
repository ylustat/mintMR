\name{mintMR}
\alias{mintMR}
\title{An integrative multi-context Mendelian randomization method for identifying risk genes across human tissues}
\description{
  The function implements the mintMR method for mapping gene expression and molecular traits as risk exposures across multiple contexts/tissues. 
}
\usage{
mintMR(gammah, Gammah, se1, se2, group, opts = NULL, corr_mat = NULL, Lambda = NULL, CC = 2, PC1 = 1, PC2 = 1, display_progress = TRUE)
}
\arguments{
  \item{gammah}{A list of estimated IV-to-exposure effects across multiple contexts or tissues, where each element corresponds to one gene.}
  \item{Gammah}{A list of estimated IV-to-outcome effect.}
  \item{se1}{A list of standard errors for the IV-to-Exposure effects.}
  \item{se2}{A list of standard errors for the IV-to-Outcome effects.}
  \item{group}{A list of indices, where each element is the vector of column indices for one exposure in `gamma_hat` and `se_g`. The input IV-to-exposure statistics contains statistics for multiple exposures, each exposure having effects from multiple contexts/tissues. The group parameter specifies which columns in each element of `gammah` correspond to the particular exposure. For instance, `group=list(exposure1=c(1,2,3), exposure2=c(4,5,6))` indicates that columns 1 to 3 in each `gammah` matrix are for exposure 1, while columns 4 to 6 are for exposure 2.}
  \item{opts}{Optional; a list of options for the choices of prior parameters in the Gibbs sampling algorithm. If not specified, default values are used.}
  \item{corr_mat}{Optional; a list of matrices corresponding to the correlation matrices among IVs. If not provided, independent IV are assumed.}
\item{Lambda}{Optional; a matrix of Lambda parameters if there is sample overlap. If not provided, non-overlapping samples are assumed.}
  \item{CC}{The number of canonical components to retain in the canonical correlation analysis (CCA). Default is 2.}
  \item{PC1}{The number of principal components to retain in the principal correlation analysis (PCA) for the first exposure. Default is 1.}
  \item{PC2}{The number of principal components to retain in the PCA for the second exposure. Default is 1.}
  \item{display_progress}{Logical; if TRUE, displays the progress of the algorithm. Default is TRUE.}
}
\value{
  Returns a list containing:
  \itemize{
    \item \code{Estimate}: A matrix of estimated causal effect sizes for each gene in each context.
    \item \code{Pvalue}: A matrix of p-values for testing non-zero effects.
  }
}
\examples{
data(example_data)
K1 <- K2 <- 5
group <- list(exposure1 = 1:5, exposure2 = 6:10)
set.seed(123)
res <- mintMR(gammah = example_data$gamma_hat,
              Gammah = example_data$Gamma_hat,
              se1 = example_data$se_g,
              se2 = example_data$se_G,
              corr_mat = example_data$LD, 
              group = group)
names(res)
}

