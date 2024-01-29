\name{mintMR}
\alias{mintMR}
\title{An integrative multi-context Mendelian randomization method for identifying risk genes across human tissues}
\description{
  Implementation of a Gibbs sampling algorithm for the mintMR method. The function is designed to map gene expression and molecular traits as risk exposures across multiple contexts/tissues.
}
\usage{
mintMR(gammah, Gammah, se1, se2, group, opts = NULL, corr_mat = NULL, Lambda = NULL, CC = 2, PC1 = 1, PC2 = 1, display_progress = TRUE)
}
\arguments{
  \item{gammah}{A list of IV-to-Exposure effect sizes for each exposure across multiple contexts.}
  \item{Gammah}{A list of IV-to-Outcome effect sizes.}
  \item{se1}{A list of standard errors for the IV-to-Exposure effects.}
  \item{se2}{A list of standard errors for the IV-to-Outcome effects.}
  \item{group}{A list of specifying the relationship between exposures and their respective contexts. This parameter specifies which columns in each element of `gammah` correspond to particular exposures. For instance, `group=list(exposure1=c(1,2,3), exposure2=c(4,5,6))` indicates that columns 1 to 3 in each `gammah` matrix are associated with exposure 1, while columns 4 to 6 are linked to exposure 2.}
  \item{opts}{Optional; a list of options for the choices of prior parameters in the Gibbs sampling algorithm. If not specified, default values are used.}
  \item{corr_mat}{Optional; a list of correlation matrices corresponding to the correlation between IVs. If not provided, independent IV are asssumed.}
  \item{Lambda}{Optional; a matrix of Lambda parameters for the case of sample overlap. If not provided, non-overlapping samples are asssumed.}
  \item{CC}{Number of canonical components to keep in canonical correlation analysis (CCA). Default is 2.}
  \item{PC1}{Numbers of principal components to keep in principal correlation analysis (PCA) for the first exposure. Default is 1.}
  \item{PC2}{Numbers of principal components to keep in PCA for the second exposure. Default is 1.}
  \item{display_progress}{Logical; if TRUE, displays the progress of the algorithm. Default is TRUE.}
}
\value{
  Returns a list containing:
  \itemize{
    \item \code{Estimate}: A matrix of estimated causal effect sizes for each gene across contexts.
    \item \code{Pvalue}: A matrix of p-values associated with the estimated effects.
  }
}
\examples{
data(example_data)
L <- 50
K1 <- K2 <- 5
group <- list(exposure1 = 1:K1, exposure2 = 1:K2+K1)
set.seed(123)
res <- mintMR(gammah = example_data$gamma_hat,
              Gammah = example_data$Gamma_hat,
              se1 = example_data$se_g,
              se2 = example_data$se_G,
              corr_mat = example_data$LD, 
              group = group)
names(res)
}