# ==============================================================================
# Credit Card Fraud Detection Using Machine Learning in R
# An Imbalanced Classification Approach  (IDS Case Study)
# ==============================================================================
# HOW TO RUN
#   1. Download "creditcard.csv" from Kaggle ("Credit Card Fraud Detection",
#      ULB Machine Learning Group) and put it in the same folder as this script
#      (or change DATA_PATH below).
#   2. In RStudio: set the working directory to this folder
#      (Session > Set Working Directory > To Source File Location) and click
#      "Source". Or from a terminal:  Rscript fraud_detection.R
#   3. All plots and tables are saved in the "outputs/" folder.
#      The console prints the final model comparison table.
#
# PIPELINE (matches the slides)
#   1. Data collection        -> load CSV, structure, missing values, duplicates
#   2. Preprocessing          -> stratified 70/30 split, scale Time and Amount
#   3. Exploratory analysis   -> class balance, Amount, Time, correlations
#   4. Imbalance handling     -> class weights | undersampling | SMOTE
#   5. Model building         -> Logistic Regression, Decision Tree,
#                                Random Forest (ranger), XGBoost
#   6. Evaluation             -> Precision, Recall, F1, ROC-AUC, PR-AUC,
#                                confusion matrix, 5-fold stratified CV
# ==============================================================================

# ---- 0. Settings -------------------------------------------------------------
set.seed(42)

DATA_PATH  <- "C:/Users/manju/Downloads/creditcard.csv"   # forward slashes on Windows!
OUT_DIR    <- "outputs"
RF_TREES   <- 200      # lower (e.g. 100) if your computer is slow
XGB_ROUNDS <- 200
RUN_CV     <- TRUE     # 5-fold stratified cross-validation (set FALSE to skip)
CV_FOLDS   <- 5
CV_STRATEGY <- "Undersampling"   # strategy used for CV (fast)

dir.create(OUT_DIR, showWarnings = FALSE)

pkgs <- c("dplyr", "tidyr", "ggplot2", "readr", "scales",
          "caret", "rpart", "ranger", "xgboost", "smotefamily",
          "pROC", "PRROC")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}
invisible(lapply(pkgs, library, character.only = TRUE))

COL_LEGIT <- "#0E9F9A"
COL_FRAUD <- "#E11D48"
theme_set(theme_minimal(base_size = 13))

save_plot <- function(p, name, w = 8, h = 5) {
  ggsave(file.path(OUT_DIR, name), p, width = w, height = h, dpi = 150)
}

# ---- 1. Data collection ------------------------------------------------------
# Robust data loader: handles "creditcard.csv is actually a folder", a ZIP that
# was never extracted, a different file name such as "creditcard (1).csv", and
# a locked file (Excel / antivirus / OneDrive) by reading from a temp copy.
locate_data <- function(path) {
  is_csv <- function(p) file.exists(p) && !dir.exists(p)
  
  if (is_csv(path)) return(path)
  
  # Folders to search: the given path (if a folder), its parent, working dir,
  # and the usual Downloads / Documents / Desktop locations.
  home <- Sys.getenv("USERPROFILE", unset = path.expand("~"))
  dirs <- unique(c(if (dir.exists(path)) path, dirname(path), getwd(),
                   file.path(home, "Downloads"), file.path(home, "Documents"),
                   file.path(home, "Desktop")))
  dirs <- dirs[dir.exists(dirs)]
  
  for (d in dirs) {
    hits <- list.files(d, pattern = "creditcard.*\\.csv$", recursive = TRUE,
                       full.names = TRUE, ignore.case = TRUE)
    hits <- hits[!dir.exists(hits)]
    if (length(hits) > 0) return(hits[1])
  }
  
  # A ZIP that was never extracted
  for (d in dirs) {
    zips <- list.files(d, pattern = "(creditcard|archive).*\\.zip$",
                       full.names = TRUE, ignore.case = TRUE)
    for (z in zips) {
      inside <- unzip(z, list = TRUE)$Name
      csv <- inside[grepl("creditcard.*\\.csv$", inside, ignore.case = TRUE)]
      if (length(csv) > 0) {
        message("Extracting ", csv[1], " from ", z)
        unzip(z, files = csv[1], exdir = tempdir())
        return(file.path(tempdir(), csv[1]))
      }
    }
  }
  
  # Last resort: let the user pick the file
  if (interactive()) {
    message("creditcard.csv not found automatically - please select it.")
    return(file.choose())
  }
  stop("Cannot find creditcard.csv. Download it from Kaggle and set DATA_PATH.")
}

data_file <- locate_data(DATA_PATH)
cat("Reading data from:", data_file, "\n")

raw <- tryCatch(
  readr::read_csv(data_file, show_col_types = FALSE),
  error = function(e) {
    # Permission denied / locked file: copy to a temp location and retry
    message("Could not open the file directly (", conditionMessage(e),
            "). Retrying from a temporary copy...")
    tmp <- tempfile(fileext = ".csv")
    ok  <- suppressWarnings(file.copy(data_file, tmp, overwrite = TRUE))
    if (!ok) {
      stop("Windows is blocking access to '", data_file, "'.\n",
           "  - Close Excel or any program that has the file open\n",
           "  - Right-click the file > Properties > tick 'Unblock'\n",
           "  - Or copy it to a simple folder such as C:/data/ and update DATA_PATH",
           call. = FALSE)
    }
    readr::read_csv(tmp, show_col_types = FALSE)
  })

cat("\n=== 1. DATA OVERVIEW ===\n")
cat("Rows:", nrow(raw), " Columns:", ncol(raw), "\n")
cat("Missing values:", sum(is.na(raw)), "\n")
n_dup <- sum(duplicated(raw))
cat("Duplicate rows:", n_dup, "(removed)\n")

df <- raw |>
  dplyr::distinct() |>
  dplyr::mutate(Class = factor(Class, levels = c(0, 1),
                               labels = c("legit", "fraud"))) |>
  as.data.frame()

class_tab <- table(df$Class)
cat("\nClass distribution (after removing duplicates):\n")
print(class_tab)
cat(sprintf("Fraud rate: %.3f%%\n", 100 * class_tab["fraud"] / sum(class_tab)))

cat("\nSummary of Time and Amount:\n")
print(summary(df[, c("Time", "Amount")]))

# ---- 2. Exploratory analysis -------------------------------------------------
cat("\n=== 2. EXPLORATORY DATA ANALYSIS (plots saved in '", OUT_DIR, "/') ===\n", sep = "")

# 2a. Class distribution (log scale, otherwise fraud is invisible)
p_class <- df |>
  dplyr::count(Class) |>
  ggplot(aes(Class, n, fill = Class)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = scales::comma(n)), vjust = -0.5) +
  scale_y_log10(labels = scales::comma, limits = c(1, 2e6)) +
  scale_fill_manual(values = c(legit = COL_LEGIT, fraud = COL_FRAUD)) +
  labs(title = "Class distribution (log scale)", x = NULL,
       y = "Number of transactions") +
  theme(legend.position = "none")
save_plot(p_class, "01_class_distribution.png", 6, 5)

# 2b. Amount: fraud vs legitimate
p_amount <- ggplot(df, aes(Class, Amount + 1, fill = Class)) +
  geom_boxplot(outlier.alpha = 0.15) +
  scale_y_log10(labels = scales::comma) +
  scale_fill_manual(values = c(legit = COL_LEGIT, fraud = COL_FRAUD)) +
  labs(title = "Transaction Amount by class (log scale)", x = NULL,
       y = "Amount + 1") +
  theme(legend.position = "none")
save_plot(p_amount, "02_amount_by_class.png", 6, 5)

cat("\nAmount summary by class:\n")
print(df |> dplyr::group_by(Class) |>
        dplyr::summarise(mean = mean(Amount), median = median(Amount),
                         max = max(Amount), .groups = "drop"))

# 2c. Time of transaction
p_time <- ggplot(df, aes(Time / 3600, fill = Class)) +
  geom_density(alpha = 0.45) +
  scale_fill_manual(values = c(legit = COL_LEGIT, fraud = COL_FRAUD)) +
  labs(title = "Transactions over time", x = "Hours since first transaction",
       y = "Density")
save_plot(p_time, "03_time_density.png", 8, 5)

# 2d. Correlation heatmap
cor_mat <- cor(dplyr::mutate(df, Class = as.numeric(Class == "fraud")))
cor_long <- as.data.frame(as.table(cor_mat))
p_cor <- ggplot(cor_long, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = COL_FRAUD, mid = "white", high = COL_LEGIT,
                       midpoint = 0, limits = c(-1, 1), name = "Corr") +
  labs(title = "Correlation heatmap", x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 7),
        axis.text.y = element_text(size = 7))
save_plot(p_cor, "04_correlation_heatmap.png", 9, 8)

cat("\nFeatures most correlated with Class (absolute value):\n")
cor_class <- sort(abs(cor_mat[, "Class"]), decreasing = TRUE)
print(round(head(cor_class[names(cor_class) != "Class"], 8), 3))

# 2e. Density of key features by class
key_feats <- c("V4", "V10", "V11", "V12", "V14", "V17")
p_dens <- df |>
  dplyr::select(Class, dplyr::all_of(key_feats)) |>
  tidyr::pivot_longer(-Class, names_to = "Feature", values_to = "Value") |>
  ggplot(aes(Value, fill = Class)) +
  geom_density(alpha = 0.45) +
  facet_wrap(~Feature, scales = "free") +
  scale_fill_manual(values = c(legit = COL_LEGIT, fraud = COL_FRAUD)) +
  labs(title = "Distribution of key features by class", x = NULL, y = "Density")
save_plot(p_dens, "05_key_feature_densities.png", 10, 6)

# ---- 3. Preprocessing: split + scaling ---------------------------------------
cat("\n=== 3. PREPROCESSING ===\n")

idx   <- as.vector(caret::createDataPartition(df$Class, p = 0.7, list = FALSE))
train <- df[idx, ]
test  <- df[-idx, ]

# Scale Time and Amount using TRAIN statistics only (avoids data leakage).
# V1-V28 are already PCA components.
sc <- list(Time   = c(mean(train$Time),   sd(train$Time)),
           Amount = c(mean(train$Amount), sd(train$Amount)))
scale_df <- function(d) {
  d$Time   <- (d$Time   - sc$Time[1])   / sc$Time[2]
  d$Amount <- (d$Amount - sc$Amount[1]) / sc$Amount[2]
  d
}
train <- scale_df(train)
test  <- scale_df(test)

predictors <- setdiff(names(train), "Class")

cat("Train:", nrow(train), "rows | fraud =", sum(train$Class == "fraud"), "\n")
cat("Test :", nrow(test),  "rows | fraud =", sum(test$Class == "fraud"), "\n")

# ---- 4. Imbalance handling ---------------------------------------------------
# Applied to TRAINING data only. The test set keeps its natural imbalance.

undersample <- function(d, ratio = 1) {
  fraud <- d[d$Class == "fraud", ]
  legit <- d[d$Class == "legit", ]
  n <- min(nrow(legit), round(nrow(fraud) * ratio))
  rbind(fraud, legit[sample(nrow(legit), n), ])
}

# SMOTE: shrink the majority class to ~20k rows, then create synthetic fraud
# cases so both classes are roughly balanced (keeps run-time reasonable).
smote_balance <- function(d, majority_n = 20000) {
  fraud   <- d[d$Class == "fraud", ]
  legit   <- d[d$Class == "legit", ]
  legit_s <- legit[sample(nrow(legit), min(nrow(legit), majority_n)), ]
  d2      <- rbind(fraud, legit_s)
  dup     <- max(1, round(nrow(legit_s) / nrow(fraud)) - 1)
  sm <- smotefamily::SMOTE(X = d2[, predictors], target = d2$Class,
                           K = 5, dup_size = dup)
  out <- sm$data
  out$Class <- factor(out$class, levels = c("legit", "fraud"))
  out$class <- NULL
  out
}

strategies <- c("Baseline (class weights)", "Undersampling", "SMOTE")

apply_strategy <- function(name, d) {
  switch(name,
         "Baseline (class weights)" = list(data = d, weighted = TRUE),
         "Undersampling"            = list(data = undersample(d), weighted = FALSE),
         "SMOTE"                    = list(data = smote_balance(d), weighted = FALSE),
         stop("Unknown strategy: ", name))
}

# ---- 5. Models ---------------------------------------------------------------
models <- c("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost")

fit_model <- function(name, d, weighted) {
  n_pos <- sum(d$Class == "fraud")
  n_neg <- sum(d$Class == "legit")
  wpos  <- if (weighted) n_neg / n_pos else 1     # fraud class weight
  w     <- ifelse(d$Class == "fraud", wpos, 1)
  
  switch(name,
         "Logistic Regression" =
           suppressWarnings(glm(Class ~ ., data = d, family = binomial(), weights = w)),
         
         "Decision Tree" =
           rpart::rpart(Class ~ ., data = d, weights = w, method = "class",
                        control = rpart::rpart.control(cp = 0.001, maxdepth = 10)),
         
         "Random Forest" =
           ranger::ranger(Class ~ ., data = d, num.trees = RF_TREES,
                          probability = TRUE, class.weights = c(1, wpos),
                          min.node.size = 5, seed = 42),
         
         "XGBoost" = {
           dtrain <- xgboost::xgb.DMatrix(data = as.matrix(d[, predictors]),
                                          label = as.numeric(d$Class == "fraud"))
           params <- list(objective = "binary:logistic", eval_metric = "aucpr",
                          max_depth = 6, eta = 0.1, subsample = 0.8,
                          colsample_bytree = 0.8, scale_pos_weight = wpos)
           xgboost::xgb.train(params = params, data = dtrain,
                              nrounds = XGB_ROUNDS, verbose = 0)
         },
         stop("Unknown model: ", name))
}

# Probability of fraud for every model
predict_prob <- function(name, fit, d) {
  switch(name,
         "Logistic Regression" =
           suppressWarnings(as.numeric(predict(fit, newdata = d, type = "response"))),
         "Decision Tree" = as.numeric(predict(fit, newdata = d, type = "prob")[, "fraud"]),
         "Random Forest" = as.numeric(predict(fit, data = d)$predictions[, "fraud"]),
         "XGBoost"       = as.numeric(predict(fit, as.matrix(d[, predictors]))))
}

# ---- 6. Evaluation helpers ---------------------------------------------------
# Accuracy is deliberately NOT used: with 0.17% fraud it is misleading.
eval_metrics <- function(truth, prob, thr = 0.5) {
  act  <- truth == "fraud"
  pred <- prob >= thr
  TP <- sum(pred & act);  FP <- sum(pred & !act)
  FN <- sum(!pred & act); TN <- sum(!pred & !act)
  precision <- if ((TP + FP) > 0) TP / (TP + FP) else 0
  recall    <- if ((TP + FN) > 0) TP / (TP + FN) else 0
  f1 <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
  roc_auc <- as.numeric(pROC::auc(pROC::roc(act, prob, levels = c(FALSE, TRUE),
                                            direction = "<", quiet = TRUE)))
  pr_auc  <- PRROC::pr.curve(scores.class0 = prob[act],
                             scores.class1 = prob[!act])$auc.integral
  data.frame(Precision = precision, Recall = recall, F1 = f1,
             ROC_AUC = roc_auc, PR_AUC = pr_auc,
             TP = TP, FP = FP, FN = FN, TN = TN)
}

# ---- 7. Train and evaluate every model x strategy ----------------------------
cat("\n=== 4-6. TRAINING AND EVALUATION (this can take a few minutes) ===\n")

results <- list()
probs   <- list()
xgb_fit <- list()

for (s in strategies) {
  cat("\n>>> Imbalance strategy:", s, "\n")
  prep <- apply_strategy(s, train)
  cat("    training rows:", nrow(prep$data),
      "| fraud:", sum(prep$data$Class == "fraud"), "\n")
  
  for (m in models) {
    t0  <- Sys.time()
    fit <- fit_model(m, prep$data, prep$weighted)
    p   <- predict_prob(m, fit, test)
    
    key <- paste(m, "|", s)
    results[[key]] <- cbind(Model = m, Strategy = s, eval_metrics(test$Class, p))
    probs[[key]]   <- p
    if (m == "XGBoost") xgb_fit[[s]] <- fit
    
    cat(sprintf("    %-20s PR-AUC = %.4f | F1 = %.4f | %.0f sec\n", m,
                results[[key]]$PR_AUC, results[[key]]$F1,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
}

results_df <- dplyr::bind_rows(results) |>
  dplyr::arrange(dplyr::desc(PR_AUC))
write.csv(results_df, file.path(OUT_DIR, "model_comparison.csv"), row.names = FALSE)

cat("\n=== MODEL COMPARISON (test set, sorted by PR-AUC) ===\n")
print(results_df |>
        dplyr::mutate(dplyr::across(Precision:PR_AUC, ~ round(.x, 4))),
      row.names = FALSE)

# ---- 8. Plots of results -----------------------------------------------------
p_cmp <- results_df |>
  tidyr::pivot_longer(c(Precision, Recall, F1, PR_AUC),
                      names_to = "Metric", values_to = "Value") |>
  dplyr::mutate(Model = factor(Model, levels = models)) |>
  ggplot(aes(Model, Value, fill = Strategy)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~Metric, ncol = 2) +
  scale_fill_manual(values = c(COL_LEGIT, "#4F46E5", COL_FRAUD)) +
  labs(title = "Model comparison on the test set", x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
save_plot(p_cmp, "06_model_comparison.png", 10, 7)

# Precision-Recall curves of the 5 best combinations
act_test <- test$Class == "fraud"
top_keys <- head(paste(results_df$Model, "|", results_df$Strategy), 5)
curves <- lapply(top_keys, function(k) {
  pr <- PRROC::pr.curve(scores.class0 = probs[[k]][act_test],
                        scores.class1 = probs[[k]][!act_test], curve = TRUE)
  cv <- as.data.frame(pr$curve[, 1:2]); names(cv) <- c("Recall", "Precision")
  cv <- cv[unique(round(seq(1, nrow(cv), length.out = min(nrow(cv), 2000)))), ]
  cv$Combination <- k
  cv
})
p_pr <- ggplot(dplyr::bind_rows(curves), aes(Recall, Precision, colour = Combination)) +
  geom_line(linewidth = 0.9) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "Precision-Recall curves (top 5 combinations)") +
  theme(legend.position = "bottom", legend.direction = "vertical")
save_plot(p_pr, "07_precision_recall_curves.png", 8, 7)

# ---- 9. Best model: confusion matrix -----------------------------------------
best <- results_df[1, ]
best_key <- paste(best$Model, "|", best$Strategy)
cat("\n=== BEST MODEL:", best_key, "===\n")

best_pred <- factor(ifelse(probs[[best_key]] >= 0.5, "fraud", "legit"),
                    levels = c("legit", "fraud"))
cm <- caret::confusionMatrix(best_pred, test$Class, positive = "fraud",
                             mode = "prec_recall")
print(cm)
capture.output(cm, file = file.path(OUT_DIR, "best_model_confusion_matrix.txt"))

# XGBoost feature importance (for the best XGBoost strategy)
best_xgb_strategy <- results_df |>
  dplyr::filter(Model == "XGBoost") |> dplyr::slice(1) |> dplyr::pull(Strategy)
imp <- xgboost::xgb.importance(feature_names = predictors,
                               model = xgb_fit[[best_xgb_strategy]])
p_imp <- imp |> head(10) |>
  ggplot(aes(reorder(Feature, Gain), Gain)) +
  geom_col(fill = COL_LEGIT) + coord_flip() +
  labs(title = paste("XGBoost top-10 features (", best_xgb_strategy, ")", sep = ""),
       x = NULL, y = "Gain")
save_plot(p_imp, "08_xgboost_feature_importance.png", 7, 5)

# ---- 10. 5-fold stratified cross-validation ----------------------------------
if (RUN_CV) {
  cat("\n=== 5-FOLD STRATIFIED CROSS-VALIDATION (", CV_STRATEGY, ") ===\n", sep = "")
  folds <- caret::createFolds(train$Class, k = CV_FOLDS, returnTrain = TRUE)
  cv_out <- list()
  for (i in seq_along(folds)) {
    cat("  fold", i, "of", CV_FOLDS, "\n")
    tr   <- train[folds[[i]], ]
    va   <- train[-folds[[i]], ]
    prep <- apply_strategy(CV_STRATEGY, tr)
    for (m in models) {
      fit <- fit_model(m, prep$data, prep$weighted)
      cv_out[[length(cv_out) + 1]] <-
        cbind(Fold = i, Model = m, eval_metrics(va$Class, predict_prob(m, fit, va)))
    }
  }
  cv_summary <- dplyr::bind_rows(cv_out) |>
    dplyr::group_by(Model) |>
    dplyr::summarise(dplyr::across(c(Precision, Recall, F1, ROC_AUC, PR_AUC),
                                   list(mean = mean, sd = sd)),
                     .groups = "drop") |>
    dplyr::arrange(dplyr::desc(PR_AUC_mean))
  write.csv(cv_summary, file.path(OUT_DIR, "cross_validation_results.csv"),
            row.names = FALSE)
  print(cv_summary |> dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, 4))),
        width = Inf)
}

# ---- 11. Wrap-up --------------------------------------------------------------
writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "session_info.txt"))

cat("\n=== DONE ===\n")
cat("Best model by PR-AUC:", best_key, "\n")
cat(sprintf("  Precision = %.3f | Recall = %.3f | F1 = %.3f | PR-AUC = %.3f | ROC-AUC = %.3f\n",
            best$Precision, best$Recall, best$F1, best$PR_AUC, best$ROC_AUC))
cat("All plots and tables are in the '", OUT_DIR, "' folder.\n", sep = "")