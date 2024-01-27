get_opts <- function(L,a_gamma=NULL,b_gamma=NULL,a_alpha=NULL,b_alpha=NULL,
                     a_beta=NULL,b_beta=NULL,a=NULL,b=NULL,maxIter=NULL,thin=NULL,burnin=NULL) {
  if(is.null(a_gamma)) {
    a_gamma = rep(0,L)
  }
  if(is.null(b_gamma)) {
    b_gamma = rep(0,L)
  }
  if(is.null(a_alpha)) {
    a_alpha = rep(0,L)
  }
  if(is.null(b_alpha)) {
    b_alpha = rep(0,L)
  }
  if(is.null(a_beta)) {
    a_beta = rep(0,L)
  }
  if(is.null(b_beta)) {
    b_beta = rep(0,L)
  }
  if(is.null(a)) {
    a = 0.1
  }
  if(is.null(b)) {
    b = 0.1
  }
  if(is.null(b_beta)) {
    b_beta <- rep(b_beta_init,L)
  }
  if(is.null(maxIter)) {
    maxIter <- 4e3
  }
  if(is.null(thin)) {
    thin <- 10
  }
  if(is.null(burnin)) {
    burnin <- 1e3
  }
  opts = list(a_gamma = a_gamma, b_gamma = b_gamma,
              a_alpha = a_alpha, b_alpha = b_alpha,
              a_beta = a_beta, b_beta = b_beta,
              a = a, b = b,
              maxIter = maxIter, thin = thin, burnin = burnin)
  return(opts)
}
