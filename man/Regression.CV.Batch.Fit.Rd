\name{Regression.CV.Batch.Fit}
\alias{Regression.CV.Batch.Fit}
\alias{predict.Regression.CV.Batch.FitObj}
\alias{plot.Regression.CV.Batch.FitObj}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
CV Batch Training and Diagnostics of Regression Base Learners
}
\description{
CV Batch Training and Diagnostics of Regression Base Learners.
}
\usage{
Regression.CV.Batch.Fit(instance.list, formula, data
  , ncores = 1, filemethod = FALSE, print.level = 1)
\method{predict}{Regression.CV.Batch.FitObj}(object, ..., ncores=1)
\method{plot}{Regression.CV.Batch.FitObj}(x, errfun=rmse.error, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{instance.list}{An object of class \code{\linkS4class{Instance.List}}, containing all combinations of base learner configurations and data partitions to perform CV batch training.}
  \item{formula}{Formula object expressing response variable and covariates.}
  \item{data}{Data frame expressing response variable and covariates.}
  \item{ncores}{Number of cores in parallel training.}
  \item{filemethod}{Boolean flag, indicating whether to save estimation objects to file or not.}
  \item{print.level}{Verbosity level.}
  \item{object}{Output of \code{Regression.CV.Batch.Fit}, object of class \code{\linkS4class{Regression.CV.Batch.FitObj}}.}
  \item{...}{Arguments passed from/to other functions.}
  \item{x}{Object of class \code{\linkS4class{Regression.CV.Batch.FitObj}}, to creates a plot from.}
  \item{errfun}{Error function used in generating plot.}
}

\value{
Function \code{Regression.CV.Batch.Fit} produces an object of class \code{\linkS4class{Regression.CV.Batch.FitObj}}. The \code{predict} method produces a matrix, whose columns each represent training-set predictions from one of the batch of base learners (in CV fashion).
}

\author{
Alireza S. Mahani, Mansour T.A. Sharabiani
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\linkS4class{Regression.CV.Batch.FitObj}}
}
\examples{
data(servo)
myformula <- class~motor+screw+pgain+vgain
myconfigs <- make.configs("knn")
perc.train <- 0.7
index.train <- sample(1:nrow(servo), size = round(perc.train*nrow(servo)))
data.train <- servo[index.train,]
data.predict <- servo[-index.train,]

parts <- generate.partitions(1, nrow(data.train))
myconfigs <- make.configs("knn")
instances <- make.instances(myconfigs, parts)

ret <- Regression.CV.Batch.Fit(instances, myformula, data.train)
newpred <- predict(ret, data.predict)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.

