# generic method for training base learners, must be implemented by each base learner
setGeneric("BaseLearner.Fit", function(object, formula, data, tmpfile=NULL, print.level=1, ...) standardGeneric("BaseLearner.Fit"))

# generic method for validation of fit objects
setGeneric("validate", function(object, newdata, ...) standardGeneric("validate"))

# virtual base class for base learner configurations
setClass("BaseLearner.Config", contains="VIRTUAL")

# base class for regression base learner configurations (classification, survival, etc to be added)
setClass("Regression.Config", contains="BaseLearner.Config")

make.configs <- function(baselearner=c("nnet","rf","svm","gbm","knn"), config.df, type="regression") {
  if (type!="regression") stop("invalid type argument")
  
  if (length(baselearner)==1) {
    if (missing(config.df)) {
      mycall <- call(paste("make.configs", baselearner, type, sep="."))
    } else {
      mycall <- call(paste("make.configs", baselearner, type, sep="."), config.df)
    }
    return (eval(mycall))
  } else {
    if (missing(config.df)) ret <- lapply(baselearner, make.configs)
    #else ret <- lapply(baselearner, make.configs, config.df=config.df)
    else stop("multiple baselearners cannot be configured using the same configuration dataframe")
    n <- length(ret)
    retf <- list()
    for (i in 1:n) {
      retf <- c(retf, ret[[i]])
    }
    return(retf)
  }
}

# creating a common container for estimation objects returned from core estimation calls
# first we convert S3 classes of estimation objects for base learners to S4 classes
setOldClass("gbm")
setOldClass("kknn")
setOldClass("nnet.formula")
setOldClass("svm.formula")
setOldClass("randomForest.formula")
# we then create a union of these estimation classes to form a common container (character is needed for filemethod=TRUE)
setClassUnion("RegressionEstObj", c("character","gbm","kknn","nnet.formula","svm.formula","randomForest.formula"))
# add similar classes for classification, survival, etc

# NULL is added for efficiency: when called within CV context, we don't need to retain pred pieces in each fold
setClassUnion("OptionalNumeric", c("NULL","numeric"))
# same argument, this time for partition field
setClassUnion("OptionalInteger", c("NULL","integer"))

# base class for output of base learner training
setClass("BaseLearner.FitObj", slots = c(config="BaseLearner.Config"), contains = "VIRTUAL")
setClass("Regression.FitObj", slots=c(est="RegressionEstObj", pred="OptionalNumeric"), contains="BaseLearner.FitObj")
# add similar classes for class, surv, etc

# class for output of base learner cv training
setClass("BaseLearner.CV.FitObj"
  , slots = c(fitobj.list="list", partition="OptionalInteger", filemethod="logical"
  ), contains="VIRTUAL"
)
Regression.CV.FitObj <- setClass("Regression.CV.FitObj", slots=c(pred="OptionalNumeric"), contains="BaseLearner.CV.FitObj")

# cross-validated training of regression base learners
# note: we cannot generalize to class, surv since the pred object will probably have a different structure
# alternatively, we would have to define generic functions such as 'allocate' and 'assemble' which must be implemented for each pred type
Regression.CV.Fit <- function(regression.config, formula, data, partition, tmpfiles=NULL, print.level=1) {
  nfolds <- max(partition)
  if (length(partition)!=nrow(data)) stop("length of fold parameter does not match number of rows in data")
  pred <- rep(NA, nrow(data))
  reg.list <- list()
  for (i in 1:nfolds) {
    if (print.level>=1) cat("processing fold", i, "of", nfolds, "\n")
    index_predict <- which(partition==i)
    data_train <- data[-index_predict,]
    data_predict <- data[index_predict,]
    regtmp <- BaseLearner.Fit(object=regression.config, formula=formula, data=data_train, tmpfile=tmpfiles[i], print.level=print.level-1)
    pred[index_predict] <- predict(regtmp, newdata=data_predict)
    regtmp@pred <- NULL # cv object has the full pred, so we don't need partial copies from each fold
    reg.list[[i]] <- regtmp
    flush.console()
  }
  ret <- Regression.CV.FitObj(fitobj.list=reg.list, pred=pred, partition=partition, filemethod=!is.null(tmpfiles))
  return (ret)
}

# prediction method for cross-validated baselearner
predict.Regression.CV.FitObj <- function(object, newdata=NULL, ...) {
  if (is.null(newdata)) return (object@pred)
  nfolds <- length(object@fitobj.list)
  ret <- rowMeans(sapply(1:nfolds, function(i) {
    predict(object@fitobj.list[[i]], newdata=newdata)
  }))
  return (ret)
}

### facilities for cv-based training and prediction over a batch of base learner instances ###

# base learner instance class (configuration and partition)
Instance <- setClass("Instance", slots=c(config="BaseLearner.Config", partid="character")) # should we switch partid type to integer?
# class containing a collection of baselearner instances
Instance.List <- setClass("Instance.List", slots=c(instances="list", partitions="matrix"))

# helper function for forming all permutations of configs and partitions into a list of instances
make.instances <- function(baselearner.configs, partitions) {
  nconfig <- length(baselearner.configs)
  nparts <- ncol(partitions)
  df <- expand.grid(config.index=1:nconfig, partition.index=1:nparts)
  ninstance <- nrow(df)
  instances <- lapply(1:ninstance, function(i) {
    Instance(config=baselearner.configs[[df$config.index[i]]], partid=colnames(partitions)[df$partition.index[i]])
  })
  ret <- Instance.List(instances=instances, partitions=partitions)
  return (ret)
}

# class for return object
setClassUnion("OptionalCharacter", c("NULL","character"))
BaseLearner.CV.Batch.FitObj <- setClass("BaseLearner.CV.Batch.FitObj"
  , slots=c(fitobj.list="list", instance.list="Instance.List", filemethod="logical"
    , tmpfiles="OptionalCharacter", tmpfiles.index.list="list"), contains="VIRTUAL")
Regression.CV.Batch.FitObj <- setClass("Regression.CV.Batch.FitObj", slots=c(pred="matrix", y="OptionalNumeric"), contains="BaseLearner.CV.Batch.FitObj")

# helper function for determining the start and end indexes of tempfiles in batch call, to be passed to each individual cv fit job
extract.tmpfiles.indexes <- function(instance.list) {
  n.instance <- length(instance.list@instances)
  partids <- sapply(1:n.instance, function(i) instance.list@instances[[i]]@partid)
  nfolds <- sapply(partids, function(partid) length(unique(instance.list@partitions[,partid])))
  index.end <- cumsum(nfolds)
  index.start <- index.end-nfolds+1
  ret <- list(start=index.start, end=index.end)
  return (ret)
}

# function for cv training of a batch of base learner jobs
Regression.CV.Batch.Fit <- function(instance.list, formula, data, ncores=1, filemethod=FALSE, print.level=1) {
  y <- regression.extract.response(formula, data)
  index.list <- extract.tmpfiles.indexes(instance.list)
  tmpfiles <- if (filemethod) tempfile(rep("file", max(index.list$end))) else NULL
  njobs <- length(instance.list@instances)
  pred <- array(NA, dim=c(nrow(data),njobs))
  if (ncores==1) {
    ret <- lapply(1:njobs, function(i) {
      myconfig <- instance.list@instances[[i]]@config
      baselearner.name <- extract.baselearner.name(myconfig)
      if (print.level>=1) cat("processing job", i, "of", njobs, "(", baselearner.name, ")\n")
      flush.console()
      rettmp <- Regression.CV.Fit(regression.config=myconfig, formula=formula, data=data
                     , partition=instance.list@partitions[,instance.list@instances[[i]]@partid]
                     , tmpfiles=tmpfiles[index.list$start[i]:index.list$end[i]]
                     , print.level=print.level-1)
      if (print.level>=1) cat("finished job", i, "of", njobs, "(", baselearner.name, ")\n")
      pred[,i] <<- rettmp@pred
      rettmp@pred <- NULL
      rettmp@partition <- NULL
      flush.console()
      return (rettmp)
    })
  } else {
    # to improve performance of dynamic scheduling, sort in descending order of expected execution time
    registerDoParallel(ncores)
    ret <- foreach(ii=1:njobs, .options.multicore=list(preschedule=FALSE)) %dopar% {
      myconfig <- instance.list@instances[[ii]]@config
      baselearner.name <- extract.baselearner.name(myconfig)
      if (print.level>=1) cat("processing job", ii, "of", njobs, "(", baselearner.name, ") ...\n")
      flush.console()
      rettmp <- Regression.CV.Fit(regression.config=myconfig, formula=formula, data=data
                     , partition=instance.list@partitions[,instance.list@instances[[ii]]@partid]
                     , tmpfiles=tmpfiles[index.list$start[ii]:index.list$end[ii]]
                     , print.level=print.level-1)
      if (print.level>=1) cat("finished job", ii, "of", njobs, "(", baselearner.name, ")\n")
      flush.console()
      return (rettmp)
    }
    if (print.level>=1) cat("binding predictions from all base learners...")
    sapply(1:njobs, function(i) {
      pred[,i] <<- ret[[i]]@pred
      ret[[i]]@pred <- NULL
      ret[[i]]@partition <- NULL
    })
    if (print.level>=1) cat("finished\n")
  }
  return (Regression.CV.Batch.FitObj(fitobj.list=ret, instance.list=instance.list
    , filemethod=!is.null(tmpfiles), tmpfiles=tmpfiles, tmpfiles.index.list=index.list, pred=pred, y=y))
}

# predict method
predict.Regression.CV.Batch.FitObj <- function(object, ..., ncores=1) {
  if (ncores==1) {
    predmat <- sapply(object@fitobj.list, function(x) {
      predict(x, ...)
    })
  } else {
    registerDoParallel(ncores)
    predmat <- unname(foreach(ii=1:length(object@fitobj.list), .combine=cbind, .options.multicore=list(preschedule=FALSE)) %dopar%
      predict(object@fitobj.list[[ii]], ...))
  }
  return (predmat)
}

# plot method
plot.Regression.CV.Batch.FitObj <- function(x, errfun=rmse.error, ...) {
  configs <- sapply(x@instance.list@instances, function(instance) instance@config)
  baselearner.names <- sapply(configs, extract.baselearner.name, "regression")
  baselearner.names.unique <- unique(baselearner.names)
  n.tot <- length(baselearner.names)
  n.unique <- length(baselearner.names.unique)
  y <- x@y
  error <- apply(x@pred, 2, errfun, y)
  pch.vec <- 1:n.unique # change this for custom styles
  c <- 1
  for (i in 1:n.unique) {
    idx <- which(baselearner.names==baselearner.names.unique[i])
    nidx <- length(idx)
    if (i==1) plot(c-1+1:nidx, error[idx], pch=pch.vec[i], xlim=c(1,n.tot), ylim=range(error)
      , xlab="Base Learner Instances", ylab="CV Error", main="Base Learner Performance")
    else points(c-1+1:nidx, error[idx], pch=pch.vec[i])
    c <- c+nidx
  }
  legend("topright", legend=c(baselearner.names.unique), lty=c(rep(0,n.unique)), pch=c(pch.vec))
}

# validate method
setMethod("validate", "Regression.CV.Batch.FitObj"
  , function(object, newdata, formula, errfun=rmse.error, make.plot=TRUE) {
    if (missing(newdata)) stop("must provide newdata argument")

    configs <- sapply(object@instance.list@instances, function(instance) instance@config)
    baselearner.names <- sapply(configs, extract.baselearner.name, "regression")
    baselearner.names.unique <- unique(baselearner.names)
    n.tot <- length(baselearner.names)
    n.unique <- length(baselearner.names.unique)
    y <- object@y
    pred <- object@pred
    newy <- regression.extract.response(formula, newdata)
    newpred <- predict(object, newdata)
    error <- apply(pred, 2, errfun, y)
    newerror <- apply(newpred, 2, errfun, newy)
    corr <- cor(error, newerror)
    corr.vec <- double(n.unique)

    pch.vec <- 1:n.unique # change this for custom styles
    for (i in 1:n.unique) {
      idx <- which(baselearner.names==baselearner.names.unique[i])
      corr.vec[i] <- cor(error[idx], newerror[idx])
      if (make.plot) {
        if (i==1) plot(error[idx], newerror[idx], pch=pch.vec[i], xlim=range(error), ylim=range(newerror)
          , xlab="CV Error", ylab="Hold-Out Error", main="Base Learner Performance")
        else points(error[idx], newerror[idx], pch=pch.vec[i])
      }
    }
    legend("topleft", legend=c(baselearner.names.unique), lty=c(rep(0,n.unique)), pch=c(pch.vec))
    
    ret <- list(errors=error, newerrors=newerror, corr=corr, corr.vec=corr.vec, baselearners=baselearner.names.unique)
    class(ret) <- "validate.Regression.Batch.FitObj"
    return (ret)
})

### facilities for non-cv-based training and prediction over a batch of base learner instances

BaseLearner.Batch.FitObj <- setClass("BaseLearner.Batch.FitObj"
  , slots=c(fitobj.list="list", config.list="list", filemethod="logical"
    , tmpfiles="OptionalCharacter"), contains="VIRTUAL")
Regression.Batch.FitObj <- setClass("Regression.Batch.FitObj", slots=c(pred="matrix", y="numeric"), contains="BaseLearner.Batch.FitObj")

# training
Regression.Batch.Fit <- function(config.list, formula, data, ncores=1, filemethod=FALSE, print.level=1) {
  y <- regression.extract.response(formula, data)
  njobs <- length(config.list)
  tmpfiles <- if (filemethod) tempfile(rep("file", njobs)) else NULL
  pred <- array(NA, dim=c(nrow(data),njobs))
  if (ncores==1) {
    ret <- lapply(1:njobs, function(i) {
      myconfig <- config.list[[i]]
      baselearner.name <- extract.baselearner.name(myconfig)
      if (print.level>=1) cat("processing job", i, "of", njobs, "(", baselearner.name, ")\n")
      flush.console()
      rettmp <- BaseLearner.Fit(object=myconfig, formula=formula, data=data
                     , tmpfile=tmpfiles[i]
                     , print.level=print.level-1)
      if (print.level>=1) cat("finished job", i, "of", njobs, "(", baselearner.name, ")\n")
      pred[,i] <<- rettmp@pred
      rettmp@pred <- NULL
      flush.console()
      return (rettmp)
    })
  } else {
    # to improve performance of dynamic scheduling, sort in descending order of expected execution time
    registerDoParallel(ncores)
    ret <- foreach(ii=1:njobs, .options.multicore=list(preschedule=FALSE)) %dopar% {
      myconfig <- config.list[[ii]]
      baselearner.name <- extract.baselearner.name(myconfig)
      if (print.level>=1) cat("processing job", ii, "of", njobs, "(", baselearner.name, ")\n")
      flush.console()
      rettmp <- BaseLearner.Fit(object=myconfig, formula=formula, data=data
                     , tmpfile=tmpfiles[ii]
                     , print.level=print.level-1)
      if (print.level>=1) cat("finished job", ii, "of", njobs, "(", baselearner.name, ")\n")
      flush.console()
      return (rettmp)
    }
    if (print.level>=1) cat("binding predictions from all base learners...")
    sapply(1:njobs, function(i) {
      pred[,i] <<- ret[[i]]@pred
      ret[[i]]@pred <- NULL
    })
    if (print.level>=1) cat("finished\n")
  }
  return (Regression.Batch.FitObj(fitobj.list=ret, config.list=config.list
    , filemethod=!is.null(tmpfiles), tmpfiles=tmpfiles, pred=pred, y=y))
}

# predict method
predict.Regression.Batch.FitObj <- function(object, ..., ncores=1) {
  if (ncores==1) {
    predmat <- sapply(object@fitobj.list, function(x) {
      predict(x, ...)
    })
  } else {
    registerDoParallel(ncores)
    predmat <- unname(foreach(ii=1:length(object@fitobj.list), .combine=cbind, .options.multicore=list(preschedule=FALSE)) %dopar%
      predict(object@fitobj.list[[ii]], ...))
  }
  return (predmat)
}

# plot method
plot.Regression.Batch.FitObj <- function(x, errfun=rmse.error, ...) {
  baselearner.names <- sapply(x@config.list, extract.baselearner.name, "regression")
  baselearner.names.unique <- unique(baselearner.names)
  n.tot <- length(baselearner.names)
  n.unique <- length(baselearner.names.unique)
  y <- x@y
  error <- apply(x@pred, 2, errfun, y)
  pch.vec <- 1:n.unique # change this for custom styles
  c <- 1
  for (i in 1:n.unique) {
    idx <- which(baselearner.names==baselearner.names.unique[i])
    nidx <- length(idx)
    if (i==1) plot(c-1+1:nidx, error[idx], pch=pch.vec[i], xlim=c(1,n.tot), ylim=range(error)
      , xlab="Base Learner Instances", ylab="Training Error", main="Base Learner Performance")
    else points(c-1+1:nidx, error[idx], pch=pch.vec[i])
    c <- c+nidx
  }
  legend("topright", legend=c(baselearner.names.unique), lty=c(rep(0,n.unique)), pch=c(pch.vec))
}

# validate method
setMethod("validate", "Regression.Batch.FitObj"
  , function(object, newdata, formula, errfun=rmse.error, make.plot=TRUE) {
    if (missing(newdata)) stop("must provide newdata argument")

    baselearner.names <- sapply(object@config.list, extract.baselearner.name, "regression")
    baselearner.names.unique <- unique(baselearner.names)
    n.tot <- length(baselearner.names)
    n.unique <- length(baselearner.names.unique)
    y <- object@y
    pred <- object@pred
    newy <- regression.extract.response(formula, newdata)
    newpred <- predict(object, newdata)
    error <- apply(pred, 2, errfun, y)
    newerror <- apply(newpred, 2, errfun, newy)
    corr <- cor(error, newerror)
    corr.vec <- double(n.unique)

    pch.vec <- 1:n.unique # change this for custom styles
    for (i in 1:n.unique) {
      idx <- which(baselearner.names==baselearner.names.unique[i])
      corr.vec[i] <- cor(error[idx], newerror[idx])
      if (make.plot) {
        if (i==1) plot(error[idx], newerror[idx], pch=pch.vec[i], xlim=range(error), ylim=range(newerror)
          , xlab="Training Error", ylab="Hold-Out Error", main="Base Learner Performance")
        else points(error[idx], newerror[idx], pch=pch.vec[i])
      }
    }
    legend("topleft", legend=c(baselearner.names.unique), lty=c(rep(0,n.unique)), pch=c(pch.vec))
    
    ret <- list(errors=error, newerrors=newerror, corr=corr, corr.vec=corr.vec, baselearners=baselearner.names.unique)
    class(ret) <- "validate.Regression.Batch.FitObj"
    return (ret)
})

print.validate.Regression.Batch.FitObj <- function(x, ...) {
  cat("average correlation between training and hold-out errors:", x$corr, "\n")
  cat("error correlation by base learner:\n")
  cat(paste(paste0(format(x$corr.vec, digits=3), " (", x$baselearners, ")"), collapse=", "), "\n")
}



