# mintMR

`mintMR` is a package implementing the integrative multi-context Mendelian randomization method for identifying risk genes across human tissues.

Installation
============

To install the development version of the `mintMR` package, please load the `devtools` package first. Note that mintMR requires the `CCA`, `Rcpp`, and `RcppArmadillo` packages. Additionally, ensure Rtools on Windows and Xcode on Mac OS/X are properly configured.

```
library(devtools)
install_github("ylustat/mintMR")
```

### Additional notes

If you encounter the following messages when installing this package on a server without admin access, please see the solutions below:

- If the error message shows 

  ```R
  > install_github("ylustat/mintMR")
  Downloading GitHub repo ylustat/mintMR@HEAD
  Error: Failed to install 'mintMR' from GitHub:
    Could not find tools necessary to compile a package
  Call `pkgbuild::check_build_tools(debug = TRUE)` to diagnose the problem.
  ```

  Please set `options(buildtools.check = function(action) TRUE )` before installation.

- If the error message shows

  ```R
  ERROR: 'configure' exists but is not executable -- see the 'R Installation and Administration Manual'
  ```

  Please follow instructions on this [page](https://vsoch.github.io/2013/install-r-packages-that-require-compilation-on-linux-without-sudo/).



Usage
=========

Please refer to the ['mintMR' vignette](https://github.com/ylustat/mintMR/blob/main/vignettes/mintMR.pdf) for a tutorial to use the `mintMR` package. 

# References

Yihao Lu, Ke Xu, Bowei Kang, Brandon L. Pierce, Fan Yang and Lin S. Chen. An integrative multi-context Mendelian randomization method for identifying risk genes across human tissues. Work in progress.

Development
===========

This package is maintained by Yihao Lu (yihaolu@uchicago.edu).