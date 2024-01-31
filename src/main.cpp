#include <RcppArmadillo.h>
#include <random>
#include <math.h>
#include <Rcpp.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <ctime>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;
// Define the function

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>


arma::mat normalize_mat(arma::mat X) {
  int p = X.n_cols;

  mat colmean = mean(X, 0);
  X.each_row() -= colmean;

  mat stddevs = stddev(X, 0, 0);
  for (int i = 0; i < p; i++) {
    if (stddevs(i) != 0) {
      X.col(i) /= stddevs[i];
    }
  }

  return X;
}

arma::mat do_call_rbind_vecs(List list_of_matrices) {
  int num_matrices = list_of_matrices.size();
  mat combined_matrix;

  for (int i = 0; i < num_matrices; i++) {
    NumericMatrix current_matrix = list_of_matrices[i];
    mat current_matrix_arma = as<mat>(current_matrix);
    combined_matrix = join_rows(combined_matrix, current_matrix_arma);
  }

  return trans(combined_matrix); // need transpose
}

arma::mat do_call_cbind_vecs(List list_of_matrices) {
  int num_matrices = list_of_matrices.size();
  mat combined_matrix;

  for (int i = 0; i < num_matrices; i++) {
    NumericMatrix current_matrix = list_of_matrices[i];
    mat current_matrix_arma = as<mat>(current_matrix);
    combined_matrix = join_cols(combined_matrix, current_matrix_arma);
  }

  return trans(combined_matrix); // need transpose
}

arma::mat up_truncate_matrix(arma::mat x) {
  double threshold = 6.9;
  x.elem(find(x > threshold)).fill(threshold);
  return x;
}

arma::mat low_truncate_matrix(arma::mat x) {
  double threshold = -9.21;
  x.elem(find(x < threshold)).fill(threshold);
  return x;
}

// [[Rcpp::export]]
List get_opts(int L,
              Nullable<NumericVector> a_gamma = R_NilValue,
              Nullable<NumericVector> b_gamma = R_NilValue,
              Nullable<NumericVector> a_alpha = R_NilValue,
              Nullable<NumericVector> b_alpha = R_NilValue,
              Nullable<NumericVector> a_beta = R_NilValue,
              Nullable<NumericVector> b_beta = R_NilValue,
              Nullable<double> a = R_NilValue,
              Nullable<double> b = R_NilValue,
              Nullable<int> maxIter = R_NilValue,
              Nullable<int> thin = R_NilValue,
              Nullable<int> burnin = R_NilValue) {

  NumericVector a_gamma_vec = a_gamma.isNotNull() ? as<NumericVector>(a_gamma) : NumericVector(L, 1.0);
  NumericVector b_gamma_vec = b_gamma.isNotNull() ? as<NumericVector>(b_gamma) : NumericVector(L, 0.0);
  NumericVector a_alpha_vec = a_alpha.isNotNull() ? as<NumericVector>(a_alpha) : NumericVector(L, 0.0);
  NumericVector b_alpha_vec = b_alpha.isNotNull() ? as<NumericVector>(b_alpha) : NumericVector(L, 0.0);
  NumericVector a_beta_vec = a_beta.isNotNull() ? as<NumericVector>(a_beta) : NumericVector(L, 1.0);
  NumericVector b_beta_vec = b_beta.isNotNull() ? as<NumericVector>(b_beta) : NumericVector(L, 0.01);

  double a_val = a.isNotNull() ? Rcpp::as<double>(a) : 0.1;
  double b_val = b.isNotNull() ? Rcpp::as<double>(b) : 0.1;
  int maxIter_val = maxIter.isNotNull() ? Rcpp::as<int>(maxIter) : 4000;
  int thin_val = thin.isNotNull() ? Rcpp::as<int>(thin) : 10;
  int burnin_val = burnin.isNotNull() ? Rcpp::as<int>(burnin) : 1000;

  return List::create(Named("a_gamma") = a_gamma_vec,
                      Named("b_gamma") = b_gamma_vec,
                      Named("a_alpha") = a_alpha_vec,
                      Named("b_alpha") = b_alpha_vec,
                      Named("a_beta") = a_beta_vec,
                      Named("b_beta") = b_beta_vec,
                      Named("a") = a_val,
                      Named("b") = b_val,
                      Named("maxIter") = maxIter_val,
                      Named("thin") = thin_val,
                      Named("burnin") = burnin_val);
}

// [[Rcpp::depends(RcppArmadillo)]]
vec mean_ignore_nan_inf(const mat& X) {
  vec col_mean = zeros<vec>(X.n_cols);
  for (unsigned int j = 0; j < X.n_cols; ++j) {
    vec col = X.col(j);
    vec finite_values = col(find_finite(col));
    if (finite_values.n_elem > 0) {
      col_mean(j) = mean(finite_values);
    } else {
      col_mean(j) = datum::nan; // set to NaN if all values are NaN or Inf
    }
  }
  return col_mean;
}
vec sd_ignore_nan_inf(const mat& X) {
  vec col_sd = zeros<vec>(X.n_cols);
  for (unsigned int j = 0; j < X.n_cols; ++j) {
    vec col = X.col(j);
    vec finite_values = col(find_finite(col));
    if (finite_values.n_elem > 0) {
      col_sd(j) = stddev(finite_values);
    } else {
      col_sd(j) = datum::nan; // set to NaN if all values are NaN or Inf
    }
  }
  return col_sd;
}

List summarize_result(const List& res) {

  List beta0res = res["beta0res"];
  List DeltaRes = res["DeltaRes"];
  List omegaRes = res["omegaRes"];
  int L = beta0res.size();
  int K = as<mat>(as<List>(as<List>(beta0res)[0])[0]).n_rows;

  // Initialize matrices to store results
  mat Estimate = mat(L, K, fill::zeros);
  mat Prob = mat(L, K, fill::zeros);
  mat Status = mat(L, K, fill::zeros);
  mat Pvalue = mat(L, K, fill::zeros);

  for (int s = 0; s < L; s++) {
    mat beta_est = do_call_rbind_vecs(as<List>(beta0res[s]));
    mat Delta_est = do_call_rbind_vecs(as<List>(DeltaRes[s]));
    mat omega_est = do_call_rbind_vecs(as<List>(omegaRes[s]));

    uvec all_zero = find(sum(Delta_est != 0, 0) == 0);
    if (all_zero.n_elem > 0) {
      rowvec beta_est_mean = mean(beta_est.cols(all_zero), 0);
      rowvec beta_est_sd = stddev(beta_est.cols(all_zero), 0, 0) / sqrt(beta_est.n_rows);
      vec pvals = 2 * (1 - normcdf(abs(beta_est_mean / beta_est_sd)).t());
      for (uword i = 0; i < all_zero.n_elem; ++i) {
        Pvalue(s, all_zero(i)) = pvals(i);
      }
    }

    beta_est.elem(find(Delta_est == 0)).fill(datum::nan);
    uvec nonzero = find(sum(Delta_est != 0, 0) > 0);
    if (nonzero.n_elem > 0) {
      rowvec beta_est_mean = mean_ignore_nan_inf(beta_est.cols(nonzero)).t();
      rowvec beta_est_sd = sd_ignore_nan_inf(beta_est.cols(nonzero)).t();
      vec pvals = 2 * (1 - normcdf(abs(beta_est_mean / beta_est_sd)).t());
      for (uword i = 0; i < nonzero.n_elem; ++i) {
        Pvalue(s, nonzero(i)) = pvals(i);
      }
    }
    Estimate.row(s) = mean_ignore_nan_inf(beta_est).t();
    // Prob.row(s) = mean(omega_est, 0);
    // Status.row(s) = mean(Delta_est, 0);
  }
  return List::create(Named("Estimate") = Estimate,
                      Named("Pvalue") = Pvalue);
}

Environment pkg = Environment::namespace_env("CCA");
Function cc = pkg["cc"];

List mintMR_LD(const List &gammah, const List &Gammah,
               const List &se1, const List &se2,
               const List corr_mat, const List group, const List &opts,
               bool display_progress=true,
               int CC = 2, int PC1 = 1, int PC2 = 1) {
  int L = gammah.length();
  int K = as<mat>(gammah[0]).n_cols; // gammah[0].n_cols;
  IntegerVector p(L);
  for (int i = 0; i < L; i++) {
    p[i] = as<mat>(gammah[i]).n_rows;
  }
  vec a_gamma = as<vec>(opts["a_gamma"]);
  vec b_gamma = as<vec>(opts["b_gamma"]);
  vec a_alpha = as<vec>(opts["a_alpha"]);
  vec b_alpha = as<vec>(opts["b_alpha"]);
  vec a_beta = as<vec>(opts["a_beta"]);
  vec b_beta = as<vec>(opts["b_beta"]);
  double aval = opts["a"];
  double bval = opts["b"];

  List a(L), b(L);
  for (int i = 0; i < L; i++) {
    a[i] = vec(K, fill::ones) * aval;
    b[i] = vec(K, fill::ones) * bval;
  }

  int maxIter = as<int>(opts["maxIter"]);
  int thin = as<int>(opts["thin"]);
  int burnin = as<int>(opts["burnin"]) - 1;

  // vec sgga2 = vec(L, fill::ones) * 0.01;
  vec sgal2 = vec(L, fill::ones) * 0.01;
  vec sgbeta2 = vec(L, fill::ones) * 0.01;
  vec xi2 = vec(L, fill::ones) * 0.01;

  vec sgal2xi2 = sgal2 % xi2;

  List beta0(L), omega(L);
  for (int i = 0; i < L; i++) {
    beta0[i] = vec(K, fill::ones) * 0.1;
    omega[i] = vec(K, fill::ones) * 0.1;
  }
  double u0 = 0.1;
  int numsave = maxIter / thin + 1;
  List Sgga2Res(L), Sgal2Res(L), Sgbeta2Res(L), Delta(L), beta0res(L), DeltaRes(L), omegaRes(L), mut(L), mu(L), mutRes(L), muRes(L), sgga2(L);
  List m0save(L), m1save(L);
  for (int i = 0; i < L; i++) {
    Delta[i] = vec(K, fill::zeros);
    mut[i] = vec(p[i], fill::ones) * 0.01;
    mu[i] = mat(p[i], K, fill::ones) * 0.01;
    sgga2[i] = vec(K, fill::ones) * 0.01;
    m0save[i] = mat(p[i], K, fill::ones) * 0.01;
    m1save[i] = mat(p[i], K, fill::ones) * 0.01;
  }

  for (int ell = 0; ell < L; ell++) {
    beta0res[ell] = List(numsave);
    omegaRes[ell] = List(numsave);
    DeltaRes[ell] = List(numsave);
    Sgga2Res[ell] = List(numsave);
    Sgal2Res[ell] = List(numsave);
    Sgbeta2Res[ell] = List(numsave);
    mutRes[ell] = List(numsave);
    muRes[ell] = List(numsave);

    for (int l = 0; l < numsave; l++) {
      as<List>(beta0res[ell])[l] = vec(K, fill::ones);
      as<List>(omegaRes[ell])[l] = vec(K, fill::ones);
      as<List>(DeltaRes[ell])[l] = vec(K, fill::ones);
      as<List>(Sgga2Res[ell])[l] = vec(K, fill::ones);
      as<List>(Sgal2Res[ell])[l] = 1;
      as<List>(Sgbeta2Res[ell])[l] = 1;
    }
  }


  List sG2(L), sg2(L), invsG2(L), invsg2(L);
  for (int i = 0; i < L; i++) {
    sG2[i] = as<mat>(se2[i]) % as<mat>(se2[i]);
    sg2[i] = as<mat>(se1[i]) % as<mat>(se1[i]);
    invsG2[i] = 1 / as<mat>(sG2[i]);
    invsg2[i] = 1 / as<mat>(sg2[i]);
  }


  List S_GRS_Ginv(L), S_gRS_ginv(L), S_GinvRS_Ginv(L), S_ginvRS_ginv(L), S_GRS_G(L), S_gRS_g(L);

  for (int i = 0; i < L; i++) {
    vec se2_i = se2[i];
    mat se1_i = se1[i];
    mat corr_mat_i = as<mat>(corr_mat[i]);

    // calculate S_GRS_Ginv
    mat S_GRS_Ginv_i = diagmat(se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GRS_Ginv[i] = wrap(S_GRS_Ginv_i);

    // calculate S_gRS_ginv
    List S_gRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_ginv_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_gRS_ginv[i] = wrap(S_gRS_ginv_i);

    // calculate S_GinvRS_Ginv
    mat S_GinvRS_Ginv_i = diagmat(1/se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GinvRS_Ginv[i] = wrap(S_GinvRS_Ginv_i);

    // calculate S_ginvRS_ginv
    List S_ginvRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_ginvRS_ginv_i[j] = diagmat(1/se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_ginvRS_ginv[i] = wrap(S_ginvRS_ginv_i);

    // calculate S_GRS_G
    mat S_GRS_G_i = diagmat(se2_i) * corr_mat_i * diagmat(se2_i);
    S_GRS_G[i] = wrap(S_GRS_G_i);

    // calculate S_gRS_g
    List S_gRS_g_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_g_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(se1_i.col(j));
    }
    S_gRS_g[i] = wrap(S_gRS_g_i);
  }

  List I(L);
  for (int i = 0; i < L; i++) {
    int ptmp = p[i];
    mat I_temp = eye<mat>(ptmp, ptmp);
    I[i] = I_temp;
  }
  int l = 0;
  int ell, iter;

  Progress pgbar((maxIter + burnin), display_progress);
  for (iter = 1; iter <= (maxIter + burnin); iter++) {
    pgbar.increment();
    if (Progress::check_abort()) {
      List res = List::create(
        _["beta0res"] = beta0res,
        _["Sgga2Res"] = Sgga2Res,
        _["Sgal2Res"] = Sgal2Res,
        _["omegaRes"] = omegaRes,
        _["DeltaRes"] = DeltaRes
      );
      return res;
    }
    for (ell = 0; ell < L; ell++) {
      vec invsgga2 = 1. / as<vec>(sgga2[ell]);
      vec invsgal2xi2 = 1. / sgal2xi2;

      // ----------------------- //
      // Parameters for Gamma
      // ----------------------- //
      mat v0t = inv(invsgal2xi2[ell] * as<mat>(I[ell]) + as<mat>(S_GinvRS_Ginv[ell]));
      mat mut1 = v0t * (diagmat(1 / as<vec>(sG2[ell])) * as<mat>(Gammah[ell]) +
        invsgal2xi2[ell] * (as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell]))));
      
      mat mut_ell = mvnrnd(mut1, v0t);
      mut[ell] = mut_ell;
      
    
      // ----------------------- //
      // Parameters for gamma
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        mat v1t;
        vec mut1;
        if (as<double>(as<List>(Delta[ell])[k]) == 1) {
          v1t = inv(invsgga2[k] * as<mat>(I[ell]) + as<mat>(as<List>(S_ginvRS_ginv[ell])[k]) + as<mat>(beta0[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * as<mat>(I[ell]));
          mut1 = v1t * (as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell])), 1) + as<mat>(beta0[ell])[k] * as<mat>(mu[ell]).col(k)) + as<mat>(invsg2[ell]).col(k) % as<mat>(gammah[ell]).col(k));
          mat mu_ell = as<mat>(mu[ell]);
          mu_ell.col(k) = mvnrnd(mut1, v1t);
          mu[ell] = mu_ell;
        } else {
          v1t = inv(invsgga2[k] * as<mat>(I[ell]) + as<mat>(as<List>(S_ginvRS_ginv[ell])[k]));
          mut1 = v1t * (as<mat>(invsg2[ell]).col(k) % as<mat>(gammah[ell]).col(k));
          mat mu_ell = as<mat>(mu[ell]);
          mu_ell.col(k) = mvnrnd(mut1, v1t);
          mu[ell] = mu_ell;
        }
      }

      // ----------------------- //
      // Update Delta;
      // ----------------------- //
      double pr0, pr1, prob;
      mat m0_ell = as<mat>(mu[ell]);
      mat m1_ell = as<mat>(mu[ell]);
      for (int k = 0; k < K; k++) {
        vec m0 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k) + as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        vec m1 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);

        pr0 = as<mat>(omega[ell])[k];
        pr1 = (1 - as<mat>(omega[ell])[k]) *
          exp(-0.5 * (accu((as<mat>(mut[ell]) - m1)%(as<mat>(mut[ell]) - m1)) -
          accu((as<mat>(mut[ell]) - m0)%(as<mat>(mut[ell]) - m0))) / sgal2xi2[ell]);

        prob = pr0 / (pr0 + pr1);
        vec Delta_ell = as<vec>(Delta[ell]);
        Delta_ell[k] = R::rbinom(1, prob);
        Delta[ell] = Delta_ell;
      }

      m0save[ell] = m0_ell;
      m1save[ell] = m1_ell;

      // ----------------------- //
      // Update beta0, beta1;
      // ----------------------- //
      mat beta0_ell = as<mat>(beta0[ell]);
      for (int k = 0; k < K; k++) {
        if (as<mat>(Delta[ell])[k] == 1) {
          double sig2b0t = 1 / (accu(as<mat>(mu[ell]).col(k) % as<mat>(mu[ell]).col(k)) / sgal2xi2[ell] + 1 / sgbeta2[ell]);
          double mub0 = accu(as<mat>(mu[ell]).col(k) %
                             (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % beta0_ell),1) + as<mat>(Delta[ell])[k] * beta0_ell[k] * as<mat>(mu[ell]).col(k))) *
                             invsgal2xi2[ell] * sig2b0t;
          beta0_ell[k] = mub0 + randn() * sqrt(sig2b0t);
        } else {
          double sig2b0t = sgbeta2[ell];
          beta0_ell[k] = randn() * sqrt(sig2b0t);
        }
      }

      beta0[ell] = beta0_ell;

      // ----------------------- //
      // Update sigma_alpha;
      // ----------------------- //
      double err0 = accu((as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)) % (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)));
      double ta_alpha = a_alpha[ell] + p[ell] / 2;
      double tb_alpha = b_alpha[ell] + err0 / (2 * xi2[ell]);
      sgal2[ell] = 1 / randg<double>(distr_param(ta_alpha,1/tb_alpha));
      // ----------------------- //
      // Update xi2
      // ----------------------- //
      double taxi2 = p[ell] / 2;
      double tbxi2 = 0.5*accu(err0)/sgal2[ell];
      xi2[ell] =  1 / randg<double>(distr_param(taxi2, 1/tbxi2));
      sgal2xi2[ell] = sgal2[ell]*xi2[ell];
      // ----------------------- //
      // Update sgga2
      // ----------------------- //
      double ta_gamma;
      double tb_gamma;
      int K1 = as<uvec>(group[0]).n_elem;
      ta_gamma = a_gamma[ell] + K1 * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell]).cols(as<uvec>(group[0]) - 1)%as<mat>(mu[ell]).cols(as<uvec>(group[0]) - 1))/2;
      double sgga2_grp1 = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      int K2 = as<uvec>(group[1]).n_elem;
      ta_gamma = a_gamma[ell] + K2 * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell]).cols(as<uvec>(group[1]) - 1)%as<mat>(mu[ell]).cols(as<uvec>(group[1]) - 1))/2;
      double sgga2_grp2 = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      vec sgga2_ell = as<vec>(sgga2[ell]);
      for (int k = 0; k < K1; k++) {
        sgga2_ell[k] = sgga2_grp1;
      }
      for (int k = 0; k < K2; k++) {
        sgga2_ell[k+K1] = sgga2_grp2;
      }
      sgga2[ell] = sgga2_ell;
      
      // ----------------------- //
      // Update sgbeta2
      // ----------------------- //
      double ta_beta = a_beta[ell] + K / 2;
      double tb_beta = b_beta[ell] + accu(as<mat>(beta0[ell])%as<mat>(beta0[ell]))/2;
      // sgbeta2[ell] = tb_beta/(ta_beta - 1);
      sgbeta2[ell] = (1 / randg<double>(distr_param(ta_beta, 1/tb_beta)));
      
      // ----------------------- //
      // Update omega
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        double at = as<mat>(a[ell])[k] + as<mat>(Delta[ell])[k];
        double bt = as<mat>(b[ell])[k] + (1 - as<mat>(Delta[ell])[k]);

        mat omega_ell = as<mat>(omega[ell]);

        omega_ell[k] = R::rbeta(at,bt);
        omega[ell] = omega_ell;
      }

      if(iter >= (int)burnin){
        if((iter - burnin) % thin == 0){
          as<List>(beta0res[ell])[l] = beta0[ell];
          as<List>(Sgga2Res[ell])[l] = sgga2[ell];
          as<List>(Sgal2Res[ell])[l] = sgal2xi2[ell];
          as<List>(Sgbeta2Res[ell])[l] = sgbeta2[ell];
          as<List>(omegaRes[ell])[l] = omega[ell];
          as<List>(DeltaRes[ell])[l] = Delta[ell];
          as<List>(mutRes[ell])[l] = mut[ell];
          as<List>(muRes[ell])[l] = mu[ell];
        }
      }


    }


    mat alpha_all = do_call_rbind_vecs(omega);
    mat colmean_alpha = mean(alpha_all, 0);

    mat alpha_all1 = alpha_all.cols(as<uvec>(group[0]) - 1);
    mat alpha_all2 = alpha_all.cols(as<uvec>(group[1]) - 1);

    mat U1 = log(alpha_all1 / (1 - alpha_all1)) - u0;
    mat U2 = log(alpha_all2 / (1 - alpha_all2)) - u0;

    U1 = up_truncate_matrix(U1);
    U2 = up_truncate_matrix(U2);
    U1 = low_truncate_matrix(U1);
    U2 = low_truncate_matrix(U2);

    mat norm_U1c = normalize_mat(U1);
    mat norm_U2c = normalize_mat(U2);
    mat colmean1c = mean(U1, 0);
    mat colmean2c = mean(U2, 0);
    mat sd1c = stddev(U1, 0, 0);
    mat sd2c = stddev(U2, 0, 0);

    List ccas = Rcpp::as<List>(cc(U1, U2));
    mat Ahat = ccas["xcoef"];
    mat Bhat = ccas["ycoef"];
    mat XX = U1 * Ahat;
    mat YY = U2 * Bhat;
    mat X_est1 = XX.cols(0, CC - 1) * pinv(Ahat.cols(0, CC - 1));
    mat X_est2 = YY.cols(0, CC - 1) * pinv(Bhat.cols(0, CC - 1));

    mat X_res1 = U1 - X_est1;
    mat X_res2 = U2 - X_est2;

    // perform PCA
    mat U;
    vec s;
    mat V;
    mat norm_U1 = normalize_mat(X_res1);
    mat norm_U2 = normalize_mat(X_res2);

    mat colmean1 = mean(X_res1, 0);
    mat colmean2 = mean(X_res2, 0);
    mat sd1 = stddev(X_res1, 0, 0);
    mat sd2 = stddev(X_res2, 0, 0);

    svd(U, s, V, norm_U1);
    mat X_red1 = U.cols(0, PC1 - 1) * diagmat(s.subvec(0, PC1 - 1)) * trans(V.cols(0, PC1 - 1));

    svd(U, s, V, normalize_mat(X_res2));
    mat X_red2 = U.cols(0, PC2 - 1) * diagmat(s.subvec(0, PC2 - 1)) * trans(V.cols(0, PC2 - 1));

    X_red1 = X_red1 * diagmat(sd1);
    for (int j = 0; j < X_red1.n_cols; j++) {
      X_red1.col(j) += colmean1[j];
    }
    X_red2 = X_red2 * diagmat(sd2);
    for (int j = 0; j < X_red2.n_cols; j++) {
      X_red2.col(j) += colmean2[j];
    }

    mat X_est = join_rows(X_est1, X_est2);
    mat X_red = join_rows(X_red1, X_red2);

    for (int ell = 0; ell < L; ell++) {
      mat current_omega;
      current_omega = 1 / (1 + exp(- X_red.row(ell) - X_est.row(ell) - u0));
      // current_omega = 1 / (1 + exp(-X_red.row(ell) - u0));
      omega[ell] = trans(current_omega);
    }


    if(iter >= (int)burnin){
      if((iter - burnin) % thin == 0){
        l += 1;
      }
    }
  }

  // List pvalue = 2*(R::pnorm(abs(bhat / se), 0, 1, 0, 0));

  List res = List::create(
    _["beta0res"] = beta0res,
    _["Sgga2Res"] = Sgga2Res,
    _["Sgal2Res"] = Sgal2Res,
    _["Sgbeta2Res"] = Sgbeta2Res,
    _["omegaRes"] = omegaRes,
    _["DeltaRes"] = DeltaRes,
    _["mutRes"] = mutRes,
    _["muRes"] = muRes
  );
  return res;
}

List mintMR_LD_Sample_Overlap(const List &gammah, const List &Gammah,
                              const List &se1, const List &se2,
                              const List corr_mat, const List group, const List &opts,
                              const arma::mat &Lambda,
                              bool display_progress=true,
                              int CC = 2, int PC1 = 1, int PC2 = 1) {
  int L = gammah.length();
  int K = as<mat>(gammah[0]).n_cols;
  IntegerVector p(L);
  for (int i = 0; i < L; i++) {
    p[i] = as<mat>(gammah[i]).n_rows;
  }
  mat Lambda_Offdiag = Lambda;
  Lambda_Offdiag.diag().zeros();

  List corr_mat_Offdiag(L);
  for (int i = 0; i < L; i++){
    mat matrix = as<mat>(corr_mat[i]);
    matrix.diag().zeros();
    corr_mat_Offdiag[i] = matrix;
  }


  vec a_gamma = as<vec>(opts["a_gamma"]);
  vec b_gamma = as<vec>(opts["b_gamma"]);
  vec a_alpha = as<vec>(opts["a_alpha"]);
  vec b_alpha = as<vec>(opts["b_alpha"]);
  vec a_beta = as<vec>(opts["a_beta"]);
  vec b_beta = as<vec>(opts["b_beta"]);
  double aval = opts["a"];
  double bval = opts["b"];
  List a(L), b(L);
  for (int i = 0; i < L; i++) {
    a[i] = vec(K, fill::ones) * aval;
    b[i] = vec(K, fill::ones) * bval;
  }
  double u0 = 0.1;
  int maxIter = as<int>(opts["maxIter"]);
  int thin = as<int>(opts["thin"]);
  int burnin = as<int>(opts["burnin"]) - 1;

  vec sgal2 = vec(L, fill::ones) * 0.01;
  vec sgbeta2 = vec(L, fill::ones) * 0.01;
  vec xi2 = vec(L, fill::ones) * 0.01;
  vec sgal2xi2 = sgal2 % xi2;

  List beta0(L), omega(L);
  for (int i = 0; i < L; i++) {
    beta0[i] = vec(K, fill::ones) * 0.1;
    omega[i] = vec(K, fill::ones) * 0.1;
  }

  int numsave = maxIter / thin + 1;
  List Sgga2Res(L), Sgal2Res(L), Sgbeta2Res(L), Delta(L), beta0res(L), DeltaRes(L), omegaRes(L), mut(L), mu(L), mutRes(L), muRes(L), sgga2(L);
  List m0save(L), m1save(L);
  for (int i = 0; i < L; i++) {
    Delta[i] = vec(K, fill::zeros);
    mut[i] = vec(p[i], fill::ones) * 0.01;
    mu[i] = mat(p[i], K, fill::ones) * 0.01;

    sgga2[i] = vec(p[i], fill::ones) * 0.01;
    m0save[i] = mat(p[i], K, fill::ones) * 0.01;
    m1save[i] = mat(p[i], K, fill::ones) * 0.01;
  }


  for (int ell = 0; ell < L; ell++) {
    beta0res[ell] = List(numsave);
    omegaRes[ell] = List(numsave);
    DeltaRes[ell] = List(numsave);
    Sgga2Res[ell] = List(numsave);
    Sgal2Res[ell] = List(numsave);
    Sgbeta2Res[ell] = List(numsave);
    mutRes[ell] = List(numsave);
    muRes[ell] = List(numsave);

    for (int l = 0; l < numsave; l++) {
      as<List>(beta0res[ell])[l] = vec(K, fill::ones);
      as<List>(omegaRes[ell])[l] = vec(K, fill::ones);
      as<List>(DeltaRes[ell])[l] = vec(K, fill::ones);
      as<List>(Sgga2Res[ell])[l] = vec(K, fill::ones);
      as<List>(Sgal2Res[ell])[l] = 1;
      as<List>(Sgbeta2Res[ell])[l] = 1;
    }
  }


  List sG2(L), sg2(L), invsG2(L), invsg2(L);
  for (int i = 0; i < L; i++) {
    sG2[i] = as<mat>(se2[i]) % as<mat>(se2[i]);
    sg2[i] = as<mat>(se1[i]) % as<mat>(se1[i]);
    invsG2[i] = 1 / as<mat>(sG2[i]);
    invsg2[i] = 1 / as<mat>(sg2[i]);
  }
  
  List S_GRS_Ginv(L), S_gRS_ginv(L), S_GinvRS_Ginv(L), S_ginvRS_ginv(L), S_GRS_G(L), S_gRS_g(L);
  
  for (int i = 0; i < L; i++) {
    vec se2_i = se2[i];
    mat se1_i = se1[i];
    mat corr_mat_i = as<mat>(corr_mat[i]);
    
    // calculate S_GRS_Ginv
    mat S_GRS_Ginv_i = diagmat(se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GRS_Ginv[i] = wrap(S_GRS_Ginv_i);
    
    // calculate S_gRS_ginv
    List S_gRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_ginv_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_gRS_ginv[i] = wrap(S_gRS_ginv_i);
    
    // calculate S_GinvRS_Ginv
    mat S_GinvRS_Ginv_i = diagmat(1/se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GinvRS_Ginv[i] = wrap(S_GinvRS_Ginv_i);
    
    // calculate S_ginvRS_ginv
    List S_ginvRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_ginvRS_ginv_i[j] = diagmat(1/se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_ginvRS_ginv[i] = wrap(S_ginvRS_ginv_i);
    
    // calculate S_GRS_G
    mat S_GRS_G_i = diagmat(se2_i) * corr_mat_i * diagmat(se2_i);
    S_GRS_G[i] = wrap(S_GRS_G_i);
    
    // calculate S_gRS_g
    List S_gRS_g_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_g_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(se1_i.col(j));
    }
    S_gRS_g[i] = wrap(S_gRS_g_i);
  }
  

  List I(L);
  for (int i = 0; i < L; i++) {
    int ptmp = p[i];
    mat I_temp = eye<mat>(ptmp, ptmp);
    I[i] = I_temp;
  }
  int l = 0;
  int ell, iter;
  Progress pgbar((maxIter + burnin), display_progress);
  for (iter = 1; iter <= (maxIter + burnin); iter++) {
    pgbar.increment();
    if (Progress::check_abort()) {
      List res = List::create(
        _["beta0res"] = beta0res,
        _["Sgga2Res"] = Sgga2Res,
        _["Sgal2Res"] = Sgal2Res,
        _["omegaRes"] = omegaRes,
        _["DeltaRes"] = DeltaRes
      );
      return res;
    }

    for (ell = 0; ell < L; ell++) {
      vec invsgga2 = 1. / as<vec>(sgga2[ell]);
      vec invsgal2xi2 = 1. / sgal2xi2;

      // ----------------------- //
      // Parameters for Gamma
      // ----------------------- //
      mat v0t = inv(invsgal2xi2[ell] * as<mat>(I[ell]) + Lambda(0,0) * as<mat>(S_GinvRS_Ginv[ell]));
      mat mut1 = (Lambda(0,0) * diagmat(1 / as<vec>(sG2[ell])) * as<mat>(Gammah[ell]) +
        invsgal2xi2[ell] * (as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell]))));
      for (int k1 = 0; k1 < K; k1++) {
        mut1 = mut1 + Lambda(0,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(gammah[ell]).col(k1);
        mut1 = mut1 - Lambda(0,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mu[ell]).col(k1);
      }
      mut1 = v0t * mut1;
      mat mut_ell = mvnrnd(mut1, v0t);
      mut[ell] = mut_ell;
      // ----------------------- //
      // Parameters for gamma
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        mat v1t;
        vec mut1;
        v1t = inv(invsgga2[k] * as<mat>(I[ell]) + Lambda(k+1,k+1) * as<mat>(as<List>(S_ginvRS_ginv[ell])[k]) + as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * as<mat>(I[ell]));
        mut1 = as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell])), 1) + as<mat>(beta0[ell])[k] * as<mat>(mu[ell]).col(k)) +
          Lambda(0,k+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(Gammah[ell])-
          Lambda(0,k+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mut[ell]);
        for (int k1 = 0; k1 < K; k1++) {
          mut1 = mut1 + Lambda(k+1,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(gammah[ell]).col(k1);
          if(k1!=k){
            mut1 = mut1 - Lambda(k+1,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(mu[ell]).col(k1);
          }
        }
        mut1 = v1t * mut1;
        mat mu_ell = as<mat>(mu[ell]);
        mu_ell.col(k) = mvnrnd(mut1, v1t);
        mu[ell] = mu_ell;
      }
      
      // ----------------------- //
      // Update Delta;
      // ----------------------- //
      double pr0, pr1, prob;
      mat m0_ell = as<mat>(mu[ell]);
      mat m1_ell = as<mat>(mu[ell]);
      for (int k = 0; k < K; k++) {
        vec m0 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k) + as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        vec m1 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);

        pr0 = as<mat>(omega[ell])[k];
        pr1 = (1 - as<mat>(omega[ell])[k]) *
          exp(-0.5 * (accu((as<mat>(mut[ell]) - m1)%(as<mat>(mut[ell]) - m1)) -
          accu((as<mat>(mut[ell]) - m0)%(as<mat>(mut[ell]) - m0))) / sgal2xi2[ell]);

        prob = pr0 / (pr0 + pr1);
        vec Delta_ell = as<vec>(Delta[ell]);
        Delta_ell[k] = R::rbinom(1, prob);

        Delta[ell] = Delta_ell;
      }

      m0save[ell] = m0_ell;
      m1save[ell] = m1_ell;


      // ----------------------- //
      // Update beta0, beta1;
      // ----------------------- //
      mat beta0_ell = as<mat>(beta0[ell]);
      for (int k = 0; k < K; k++) {
        if (as<mat>(Delta[ell])[k] == 1) {
          double sig2b0t = 1 / (accu(as<mat>(mu[ell]).col(k) % as<mat>(mu[ell]).col(k)) / sgal2xi2[ell] + 1 / sgbeta2[ell]);
          double mub0 = accu(as<mat>(mu[ell]).col(k) %
                             (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % beta0_ell),1) + as<mat>(Delta[ell])[k] * beta0_ell[k] * as<mat>(mu[ell]).col(k))) *
                             invsgal2xi2[ell] * sig2b0t;
          beta0_ell[k] = mub0 + randn() * sqrt(sig2b0t);
        } else {
          double sig2b0t = sgbeta2[ell];
          beta0_ell[k] = randn() * sqrt(sig2b0t);
        }
      }


      beta0[ell] = beta0_ell;

      // ----------------------- //
      // Update sigma_alpha;
      // ----------------------- //
      double err0 = accu((as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)) % (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)));
      double ta_alpha = a_alpha[ell] + p[ell] / 2;
      double tb_alpha = b_alpha[ell] + err0 / (2 * xi2[ell]);

      sgal2[ell] = 1 / randg<double>(distr_param(ta_alpha,1/tb_alpha));

      // ----------------------- //
      // Update xi2
      // ----------------------- //
      double taxi2 = p[ell] / 2;
      double tbxi2 = 0.5*accu(err0)/sgal2[ell];
      xi2[ell] =  1 / randg<double>(distr_param(taxi2, 1/tbxi2));
      sgal2xi2[ell] = sgal2[ell]*xi2[ell];

      // ----------------------- //
      // Update sgga2
      // ----------------------- //
      double ta_gamma;
      double tb_gamma;
      int K1 = as<uvec>(group[0]).n_elem;
      ta_gamma = a_gamma[ell] + K1 * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell]).cols(as<uvec>(group[0]) - 1)%as<mat>(mu[ell]).cols(as<uvec>(group[0]) - 1))/2;
      double sgga2_grp1 = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      int K2 = as<uvec>(group[1]).n_elem;
      ta_gamma = a_gamma[ell] + K2 * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell]).cols(as<uvec>(group[1]) - 1)%as<mat>(mu[ell]).cols(as<uvec>(group[1]) - 1))/2;
      double sgga2_grp2 = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      vec sgga2_ell = as<vec>(sgga2[ell]);
      for (int k = 0; k < K1; k++) {
        sgga2_ell[k] = sgga2_grp1;
      }
      for (int k = 0; k < K2; k++) {
        sgga2_ell[k+K1] = sgga2_grp2;
      }
      sgga2[ell] = sgga2_ell;

      // ----------------------- //
      // Update sgbeta2
      // ----------------------- //
      double ta_beta = a_beta[ell] + K / 2;
      double tb_beta = b_beta[ell] + accu(as<mat>(beta0[ell])%as<mat>(beta0[ell]))/2;
      // sgbeta2[ell] = tb_beta/(ta_beta - 1);
      sgbeta2[ell] = (1 / randg<double>(distr_param(ta_beta, 1/tb_beta)));
      
      // ----------------------- //
      // Update omega
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        double at = as<mat>(a[ell])[k] + as<mat>(Delta[ell])[k];
        double bt = as<mat>(b[ell])[k] + (1 - as<mat>(Delta[ell])[k]);

        mat omega_ell = as<mat>(omega[ell]);

        omega_ell[k] = R::rbeta(at,bt);
        omega[ell] = omega_ell;
      }

      if(iter >= (int)burnin){
        if((iter - burnin) % thin == 0){
          as<List>(beta0res[ell])[l] = beta0[ell];
          as<List>(Sgga2Res[ell])[l] = sgga2[ell];
          as<List>(Sgal2Res[ell])[l] = sgal2xi2[ell];
          as<List>(Sgbeta2Res[ell])[l] = sgbeta2[ell];
          as<List>(omegaRes[ell])[l] = omega[ell];
          as<List>(DeltaRes[ell])[l] = Delta[ell];
        }
      }
    }

    mat alpha_all = do_call_rbind_vecs(omega);
    mat colmean_alpha = mean(alpha_all, 0);


    mat alpha_all1 = alpha_all.cols(as<uvec>(group[0]) - 1);
    mat alpha_all2 = alpha_all.cols(as<uvec>(group[1]) - 1);

    mat U1 = log(alpha_all1 / (1 - alpha_all1)) - u0;
    mat U2 = log(alpha_all2 / (1 - alpha_all2)) - u0;

    U1 = up_truncate_matrix(U1);
    U2 = up_truncate_matrix(U2);
    U1 = low_truncate_matrix(U1);
    U2 = low_truncate_matrix(U2);

    mat norm_U1c = normalize_mat(U1);
    mat norm_U2c = normalize_mat(U2);
    mat colmean1c = mean(U1, 0);
    mat colmean2c = mean(U2, 0);
    mat sd1c = stddev(U1, 0, 0);
    mat sd2c = stddev(U2, 0, 0);

    List ccas = Rcpp::as<List>(cc(U1, U2));
    mat Ahat = ccas["xcoef"];
    mat Bhat = ccas["ycoef"];
    mat XX = U1 * Ahat;
    mat YY = U2 * Bhat;
    mat X_est1 = XX.cols(0, CC - 1) * pinv(Ahat.cols(0, CC - 1));
    mat X_est2 = YY.cols(0, CC - 1) * pinv(Bhat.cols(0, CC - 1));

    mat X_res1 = U1 - X_est1;
    mat X_res2 = U2 - X_est2;

    // perform PCA
    mat U;
    vec s;
    mat V;
    mat norm_U1 = normalize_mat(X_res1);
    mat norm_U2 = normalize_mat(X_res2);

    mat colmean1 = mean(X_res1, 0);
    mat colmean2 = mean(X_res2, 0);
    mat sd1 = stddev(X_res1, 0, 0);
    mat sd2 = stddev(X_res2, 0, 0);

    svd(U, s, V, norm_U1);
    mat X_red1 = U.cols(0, PC1 - 1) * diagmat(s.subvec(0, PC1 - 1)) * trans(V.cols(0, PC1 - 1));

    svd(U, s, V, normalize_mat(X_res2));
    mat X_red2 = U.cols(0, PC2 - 1) * diagmat(s.subvec(0, PC2 - 1)) * trans(V.cols(0, PC2 - 1));

    X_red1 = X_red1 * diagmat(sd1);
    for (int j = 0; j < X_red1.n_cols; j++) {
      X_red1.col(j) += colmean1[j];
    }
    X_red2 = X_red2 * diagmat(sd2);
    for (int j = 0; j < X_red2.n_cols; j++) {
      X_red2.col(j) += colmean2[j];
    }

    mat X_est = join_rows(X_est1, X_est2);
    mat X_red = join_rows(X_red1, X_red2);

    for (int ell = 0; ell < L; ell++) {
      mat current_omega;
      current_omega = 1 / (1 + exp(- X_red.row(ell) - X_est.row(ell) - u0));
      // current_omega = 1 / (1 + exp(-X_red.row(ell) - u0));
      omega[ell] = trans(current_omega);
    }

    if(iter >= (int)burnin){
      if((iter - burnin) % thin == 0){
        l += 1;
      }
    }
  }

  List res = List::create(
    _["beta0res"] = beta0res,
    _["Sgga2Res"] = Sgga2Res,
    _["Sgal2Res"] = Sgal2Res,
    _["Sgbeta2Res"] = Sgbeta2Res,
    _["omegaRes"] = omegaRes,
    _["DeltaRes"] = DeltaRes,
    _["mutRes"] = mutRes,
    _["muRes"] = muRes
  );
  return res;
}


// [[Rcpp::export]]
List mintMR(const List &gammah, const List &Gammah,
            const List &se1, const List &se2,
            const List group,
            Nullable<List> opts = R_NilValue,//const List &opts,
            Nullable<List> corr_mat = R_NilValue,
            Nullable<arma::mat> Lambda = R_NilValue,
            int CC = 2, int PC1 = 1, int PC2 = 1,
            bool display_progress = true) {

  int L = gammah.size();
  List corr_mat_list(L), opts_list(L);
  bool overlapped = Lambda.isNotNull();

  if (opts.isNull()) {
    opts_list = get_opts(L);
  } else {
    opts_list = as<List>(opts.get());
  }

  if (corr_mat.isNull()) {
    for (int i = 0; i < L; ++i) {
      int n = as<mat>(gammah[i]).n_rows;
      corr_mat_list[i] = eye<mat>(n, n);
    }
  } else {
    corr_mat_list = as<List>(corr_mat.get());
  }

  List res;
  if(overlapped) {
    arma::mat lambda_mat = as<arma::mat>(Lambda.get());
    res = mintMR_LD_Sample_Overlap(gammah, Gammah, se1, se2, corr_mat_list, group, opts_list, lambda_mat, display_progress, CC, PC1, PC2);
  } else {
    res = mintMR_LD(gammah, Gammah, se1, se2, corr_mat_list, group, opts_list, display_progress, CC, PC1, PC2);
  }

  List summary = summarize_result(res);

  return summary;
}

