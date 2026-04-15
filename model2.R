"""
Bayesian Model: Extreme Weather and Electric Demand Analysis
DS 4420 Project | Rhea Johnson & Iba Baig
"""

# packages <- c("mvnfast", "MCMCpack", "scales", "coda", "tidyverse", "lubridate")

dir.create("data",  showWarnings = FALSE)
dir.create("plots", showWarnings = FALSE)

set.seed(42)

raw <- read_csv("/Users/ibabaig/Documents/school/ML/MLProject/ma_demand_with_storms.csv", show_col_types = FALSE) %>%
  mutate(
    # debug: fix hour_of_day UTC format
    datetime_utc = as.POSIXct(datetime_utc, format = "%Y-%m-%dT%H:%M:%OSZ", tz = "UTC"),
    datetime_et  = with_tz(datetime_utc, tzone = "America/New_York")
  ) %>%
  arrange(datetime_utc)
 
# Data format check
hour_check <- table(hour(raw$datetime_et))
cat("Rows:", nrow(raw), "\n")
cat("Unique hours:", length(hour_check),"\n")
cat("Storm hour:", sum(raw$is_storm, na.rm = TRUE), "\n")
cat("Date range:", format(min(raw$datetime_et)), "to", format(max(raw$datetime_et)), "\n\n")


df <- raw %>%
  mutate(
    # Calendar features
    hour_of_day = hour(datetime_et),
    day_of_week = wday(datetime_et), # 1 = Sunday
    is_weekend = as.integer(day_of_week %in% c(1, 7)),
  
    log_demand = log(demand_mwh),
    
    # Autoregressive lags
    demand_lag24  = lag(demand_mwh, 24),
    demand_lag168 = lag(demand_mwh, 168),
    
    # Storm covariates
    storm_gust_mph = replace_na(as.numeric(storm_gust_mph), 0),
    storm_severity = replace_na(as.numeric(storm_severity), 0),
    is_snow_storm  = replace_na(as.integer(is_snow_storm),  0L),
    is_cold_event  = replace_na(as.integer(is_cold_event),  0L),
    is_wind_event  = replace_na(as.integer(is_wind_event),  0L)
  ) %>%
  filter(!is.na(demand_lag168)) %>%
  arrange(datetime_utc)

cat("Rows after feature engineering:", nrow(df), "\n")
cat("Hour distribution:\n")
print(table(df$hour_of_day))

# Train/Test Split
TRAIN_END <- as.POSIXct("2025-11-30 23:59:59", tz = "UTC")

train <- df %>% filter(datetime_utc <= TRAIN_END)
test  <- df %>% filter(datetime_utc >  TRAIN_END)

cat("Train:", nrow(train),"hours ",format(min(train$datetime_et)), format(max(train$datetime_et)),"\n")
cat("Test :", nrow(test), "hours ", format(min(test$datetime_et)), format(max(test$datetime_et)),"\n")
cat("Train storm hours:", sum(train$is_storm), "\n")
cat("Test storm hours:", sum(test$is_storm),  "\n")


make_Xy <- function(data, cols_to_keep = NULL) {
  lag24_scaled  <- as.numeric(scale(log(data$demand_lag24  + 1)))
  lag168_scaled <- as.numeric(scale(log(data$demand_lag168 + 1)))
  gust_scaled   <- as.numeric(scale(data$storm_gust_mph))
  sev_scaled    <- as.numeric(scale(data$storm_severity))
  
  # Hour_of_day
  hour_factor  <- factor(data$hour_of_day, levels = 0:23)
  hour_dummies <- model.matrix(~ hour_factor)[, -1, drop = FALSE]
  colnames(hour_dummies) <- paste0("hr", 1:23)
  
  X <- cbind(
    bias       = 1,
    lag24      = lag24_scaled,
    lag168     = lag168_scaled,
    gust       = gust_scaled,
    severity   = sev_scaled,
    is_snow    = as.numeric(data$is_snow_storm),
    is_cold    = as.numeric(data$is_cold_event),
    is_wind    = as.numeric(data$is_wind_event),
    is_weekend = as.numeric(data$is_weekend),
    hour_dummies
  )
  
  if (is.null(cols_to_keep)) {
    qr_decomp  <- qr(X)
    keep       <- qr_decomp$pivot[1:qr_decomp$rank]
    X_clean    <- X[, keep, drop = FALSE]
    dropped    <- setdiff(colnames(X), colnames(X_clean))
    if (length(dropped) > 0)
      cat("  Dropped linearly dependent columns:",
          paste(dropped, collapse = ", "), "\n")
  } else {
    # Test set
    missing_cols <- setdiff(cols_to_keep, colnames(X))
    if (length(missing_cols) > 0) {
      cat(" Test set missing columns (filling with 0):",
          paste(missing_cols, collapse = ", "), "\n")
      zero_fill <- matrix(0, nrow = nrow(X), ncol = length(missing_cols),
                          dimnames = list(NULL, missing_cols))
      X <- cbind(X, zero_fill)
    }
    X_clean <- X[, cols_to_keep, drop = FALSE]
  }
  
  list(X = X_clean, y = data$log_demand,
       kept_cols = colnames(X_clean))
}

train_Xy  <- make_Xy(train)
kept_cols <- train_Xy$kept_cols

test_Xy   <- make_Xy(test, cols_to_keep = kept_cols)

X_train   <- train_Xy$X
y_train   <- train_Xy$y
X_test    <- test_Xy$X
y_test    <- test_Xy$y

n <- nrow(X_train)
p <- ncol(X_train)

cat("Training obs (n):", n, "\n")
cat("Predictors (p) :", p, "\n")
cat("X_train cols:", ncol(X_train), "\n")
cat("X_test cols:", ncol(X_test),  "\n")
cat("Columns match:", identical(colnames(X_train), colnames(X_test)), "\n\n")

# OLS
w_hat <- solve(t(X_train) %*% X_train) %*% t(X_train) %*% y_train

cat("OLS key coefficients:\n")
for (nm in c("bias","lag24","lag168","is_cold","is_snow","is_wind","gust","severity")) {
  if (nm %in% rownames(w_hat)) {
    cat(sprintf("    %-12s : %8.4f\n", nm, w_hat[nm, 1]))
  }
}

y_hat_ols <- X_train %*% w_hat
cat("OLS training R2 Value:", round(cor(y_hat_ols, y_train)^2, 4), "\n\n")


# Prior
prior_w <- matrix(0, nrow = p, ncol = 1)
rownames(prior_w) <- kept_cols

# Informative means = key predictors
if ("bias"   %in% kept_cols) prior_w["bias",] <- mean(y_train)
if ("lag24"  %in% kept_cols) prior_w["lag24",] <- 0.5
if ("lag168" %in% kept_cols) prior_w["lag168",] <- 0.3
if ("is_cold" %in% kept_cols) prior_w["is_cold",] <- 0.05
if ("is_snow" %in% kept_cols) prior_w["is_snow",] <- 0.05

# Prior variances
prior_vars <- setNames(rep(1.0, p), kept_cols)
if ("bias"   %in% kept_cols) prior_vars["bias"] <- 2.0
if ("lag24"  %in% kept_cols) prior_vars["lag24"]  <- 0.5
if ("lag168" %in% kept_cols) prior_vars["lag168"] <- 0.5
hr_cols <- kept_cols[grepl("^hr", kept_cols)]
if (length(hr_cols) > 0) prior_vars[hr_cols] <- 0.25

prior_Sigma <- diag(prior_vars)

# Inverse gamma hyperparameters for sigma squared
prior_alpha_ig <- 3
prior_beta_ig  <- 1

cat("prior_w dimensions:", dim(prior_w), "\n")
cat("prior_sigma dim:", dim(prior_Sigma), "\n")
cat("X_train dim:", dim(X_train), "\n")
cat("p =", p, "\n")


# Gibbs Sampling
# w | sigma^2, y, X ~ MultivariateNormal(M, V)
# sigma^2 | w, y, X ~ InverseGamma(A, B)

no_samples <- 5000
burn_in <- 1000

gibbs_w <- matrix(0, nrow = no_samples, ncol = p)
colnames(gibbs_w) <- kept_cols
gibbs_sigma2 <- numeric(no_samples)

# Known values
prior_Sigma_inv <- solve(prior_Sigma)
tXX <- t(X_train) %*% X_train
tXy <- t(X_train) %*% y_train

# OLS estimate init
w_tilde <- w_hat
sigma2_tilde <- var(as.numeric(y_train - X_train %*% w_hat))

cat("n =", n, ", p =", p, "\n")

pb <- txtProgressBar(min = 0, max = no_samples, style = 3)

for (i in 1:no_samples) {
  # sample w | sigma^2, y, X
  V <- solve((1/sigma2_tilde) * tXX + prior_Sigma_inv)
  M <- V %*% ((1/sigma2_tilde) * tXy + prior_Sigma_inv %*% prior_w)
  w_tilde <- t(rmvn(1, mu = as.numeric(M), sigma = V))
  
  # sample sigma^2 | w, y, X
  resid <- y_train - X_train %*% w_tilde
  A <- (2 * prior_alpha_ig + n) / 2
  B <- (2 * prior_beta_ig + as.numeric(t(resid) %*% resid)) / 2
  sigma2_tilde <- rinvgamma(1, shape = A, scale = B)
  
  gibbs_w[i, ] <- as.numeric(w_tilde)
  gibbs_sigma2[i] <- sigma2_tilde
  
  setTxtProgressBar(pb, i)
}
close(pb)

w_post      <- gibbs_w[(burn_in + 1):no_samples, ]
sigma2_post <- gibbs_sigma2[(burn_in + 1):no_samples]
n_post      <- nrow(w_post)

cat("posterior after burn-in:", n_post, "\n")

key_params <- intersect(
  c("bias","lag24","lag168","gust","severity","is_snow","is_cold","is_wind","is_weekend"),
  kept_cols
)

for (nm in key_params) {
  ci <- quantile(w_post[, nm], c(0.025, 0.25, 0.50, 0.75, 0.975))
  cat(sprintf("  %-14s | mean=%8.4f | sd=%6.4f | 95%% CI: [%7.4f, %7.4f]\n",
              nm,
              mean(w_post[, nm]),
              sd(w_post[, nm]),
              ci[1], ci[5]))
}
ci_s <- quantile(sigma2_post, c(0.025, 0.975))
cat(sprintf("  %-14s | mean=%8.4f | sd=%6.4f | 95%% CI: [%7.4f, %7.4f]\n",
            "sigma2",
            mean(sigma2_post),
            sd(sigma2_post),
            ci_s[1], ci_s[2]))


# Convergence
check_convergence <- function(chain, name) {
  half   <- length(chain) %/% 2
  ratio  <- var(chain[(half+1):length(chain)]) / var(chain[1:half])
  cat(sprintf("  %-14s | mean=%8.4f | sd=%7.4f | var_ratio=%.3f %s\n",
              name, mean(chain), sd(chain), ratio,
              ifelse(abs(ratio - 1) < 0.15, "converged")))
}

for (nm in key_params) check_convergence(w_post[, nm], nm)
check_convergence(sigma2_post, "sigma²")

cat("\n Posterior Means vs. OLS \n")
for (nm in key_params) {
  cat(sprintf("  %-14s - Bayes: %8.4f - OLS: %8.4f\n",
              nm, mean(w_post[, nm]), w_hat[nm, 1]))
}

dev.new()
par(mfrow=c(3,2), mar=c(3,3,2,1))

trace_params <- c("bias","lag24","is_snow","is_wind","gust","sigma2")

for (nm in trace_params) {
  if (nm != "sigma2" && !(nm %in% colnames(w_post))) {
    plot.new()
    title(main=paste("Trace:", nm))
    next
  }
  chain <- if (nm == "sigma2") sigma2_post else w_post[, nm]
  plot(chain, type="l",
       main=paste("Trace:", ifelse(nm=="sigma2","σ²",nm)),
       ylab="Value", xlab="Iteration", col="#1f77b4")
  abline(h=mean(chain), col="red", lwd=1.5, lty=2)
}
par(mfrow=c(1,1))


# ACF plots
png("/Users/ibabaig/Documents/school/ML/MLProject/plots/MA_05_acf_plots.png", width=1200, height=900, res=120)
par(mfrow=c(3,2), mar=c(3,3,2,1))
for (nm in trace_params) {
  chain <- if (nm == "sigma2") sigma2_post else w_post[, nm]
  acf(chain, main=paste("ACF:", ifelse(nm=="sigma2","σ²",nm)), lag.max=50)
}
par(mfrow=c(1,1))

# Posterior Distributions
png("/Users/ibabaig/Documents/school/ML/MLProject/plots/MA_06_posterior_histograms.png", width=1400, height=1000, res=120)
par(mfrow=c(3,3), mar=c(4,3,3,1))

hist_params <- list(
  list(nm="bias", title="Intercept"),
  list(nm="lag24", title="AR: Lag-24h"),
  list(nm="lag168", title="AR: Lag-168h (week)"),
  list(nm="is_cold", title="Cold Event Premium"),
  list(nm="is_snow", title="Snow Storm Premium"),
  list(nm="is_wind", title="Wind Event Effect"),
  list(nm="gust", title="Wind Gust (mph)"),
  list(nm="severity", title="Storm Severity Rank"),
  list(nm="sigma2", title="Residual Variance σ²")
)

for (hp in hist_params) {
  chain <- if (hp$nm == "sigma2") sigma2_post else {
    if (hp$nm %in% kept_cols) w_post[, hp$nm] else rep(NA, n_post)
  }
  if (!all(is.na(chain))) {
    hist(chain, main=hp$title, xlab="Value",
         col="#aec7e8", border="white", breaks=40)
    abline(v=mean(chain), col="red",   lwd=2)
    abline(v=0, col="black", lwd=1, lty=2)
  }
}
par(mfrow=c(1,1))
dev.off()

cat("\n 95% Posterior Intervals \n")
for (nm in key_params) {
  ci <- quantile(w_post[, nm], c(0.025, 0.975))
  cat(sprintf("%-14s mean=%8.4f, 95%% CI: [%7.4f, %7.4f] %s\n",
              nm, mean(w_post[, nm]), ci[1], ci[2],
              ifelse(ci[1] > 0 | ci[2] < 0, "", "")))
}

# Posterior predictive Distribution
cat("  (", n_post, "draws ×", nrow(test), "test hours)\n\n")

mu_pred_matrix <- w_post %*% t(X_test)   
# find dims [n_post × N_test] log-scale means

noise_matrix <- matrix(
  rnorm(n_post * nrow(test), mean=0, sd=sqrt(sigma2_post)),
  nrow = n_post,
  ncol = nrow(test)
)

y_pred_log <- mu_pred_matrix + noise_matrix
y_pred_mwh <- exp(y_pred_log)

test <- test %>%
  mutate(
    pred_median = apply(y_pred_mwh, 2, median),
    pred_lo90 = apply(y_pred_mwh, 2, quantile, 0.05),
    pred_hi90  = apply(y_pred_mwh, 2, quantile, 0.95),
    pred_lo50  = apply(y_pred_mwh, 2, quantile, 0.25),
    pred_hi50  = apply(y_pred_mwh, 2, quantile, 0.75)
  )


first_storm_idx <- which(test$is_storm == 1)[1]
if (!is.na(first_storm_idx)) {
  cat("First Storm Hour:", format(test$datetime_et[first_storm_idx]), "\n")
  cat("Storm:", as.character(test$storm_label[first_storm_idx]), "\n")
  cat("Actual demand:", round(test$demand_mwh[first_storm_idx]), "MWh\n")
  cat("Post. median:", round(test$pred_median[first_storm_idx]), "MWh\n")
  cat("90% CI: [", round(test$pred_lo90[first_storm_idx]), ",",
      round(test$pred_hi90[first_storm_idx]), "] MWh\n\n")
  
  png("/Users/ibabaig/Documents/school/ML/MLProject/plots/MA_07_storm_hour_postpred.png", width=700, height=450, res=120)
  hist(y_pred_mwh[, first_storm_idx],
       main  = paste("Post. Predictive Dist ",
                     format(test$datetime_et[first_storm_idx], "%b %d %H:00")),
       xlab  = "Predicted Demand (MWh)",
       col   = "#aec7e8", border="white", breaks=50)
  abline(v=test$demand_mwh[first_storm_idx],  col="red",      lwd=2)
  abline(v=test$pred_median[first_storm_idx], col="#1f77b4",  lwd=2, lty=2)
  legend("topright",
         legend=c("Actual demand","Posterior median"),
         col=c("red","#1f77b4"), lwd=2, lty=c(1,2))
  dev.off()
}

write_csv(test, "/Users/ibabaig/Documents/school/ML/MLProject/data/ma_test_predictions.csv")

# Evaluation
rmse_fn <- function(a,p)      sqrt(mean((a-p)^2,     na.rm=TRUE))
mae_fn <- function(a,p)      mean(abs(a-p),         na.rm=TRUE)
mape_fn <- function(a,p)      mean(abs((a-p)/a)*100, na.rm=TRUE)
icr_fn <- function(a,lo,hi)  mean(a>=lo & a<=hi,    na.rm=TRUE)
miw_fn <- function(lo,hi)    mean(hi-lo,            na.rm=TRUE)
winkler_fn <- function(a,lo,hi,alpha=0.10) {
  mean((hi-lo) + ifelse(a<lo,(2/alpha)*(lo-a),
                        ifelse(a>hi,(2/alpha)*(a-hi),0)), na.rm=TRUE)
}

score <- function(d, label) {
  tibble(
    subset = label,
    n  = nrow(d),
    RMSE_MWh = rmse_fn(d$demand_mwh, d$pred_median),
    MAE_MWh  = mae_fn( d$demand_mwh, d$pred_median),
    MAPE_pct = mape_fn(d$demand_mwh, d$pred_median),
    ICR_90  = icr_fn( d$demand_mwh, d$pred_lo90, d$pred_hi90),
    MIW_90_MWh = miw_fn( d$pred_lo90,  d$pred_hi90),
    Winkler_90 = winkler_fn(d$demand_mwh, d$pred_lo90, d$pred_hi90)
  )
}

metrics <- bind_rows(
  score(test, "All test hours"),
  score(filter(test, is_storm==0), "Non-storm hours"),
  score(filter(test, is_storm==1), "Storm hours")
)

if (sum(test$is_storm) > 0) {
  snow_h <- filter(test, is_storm==1, str_detect(storm_category, "Snow"))
  cold_h <- filter(test, is_storm==1, str_detect(storm_category, "Cold"))
  wind_h <- filter(test, is_storm==1, str_detect(storm_category, "Wind"))
  if (nrow(snow_h)>0) metrics <- bind_rows(metrics, score(snow_h,"Snow storm hours"))
  if (nrow(cold_h)>0) metrics <- bind_rows(metrics, score(cold_h,"Cold event hours"))
  if (nrow(wind_h)>0) metrics <- bind_rows(metrics, score(wind_h,"Wind event hours"))
}

print(metrics %>% mutate(across(where(is.numeric), ~round(.,3))), width=120)

write_csv(metrics, "/Users/ibabaig/Documents/school/ML/MLProject/results/ma_eval_metrics.csv")

storm_windows <- read_csv("data/ma_storm_windows.csv", show_col_types=FALSE) %>%
  mutate(
    begin_et = as.POSIXct(
      as.POSIXct(begin_utc, format="%Y-%m-%d %H:%M:%S", tz="UTC"),
      tz="America/New_York"),
    end_et = as.POSIXct(
      as.POSIXct(end_utc, format="%Y-%m-%d %H:%M:%S", tz="UTC"),
      tz="America/New_York"),
    primary_cat = case_when(
      str_detect(event_cats,"Snow") ~ "Snow",
      str_detect(event_cats,"Cold") ~ "Cold",
      str_detect(event_cats,"Wind") ~ "Wind",
      TRUE ~ "Other"
    )
  )

color_map <- c("Snow"="steelblue","Cold"="#2166ac","Wind"="#d73027","Other"="gray60")

# Plot 1 = Time series
p1 <- ggplot(df, aes(x=datetime_et, y=demand_mwh)) +
  geom_rect(data=storm_windows,
            aes(xmin=begin_et,xmax=end_et,ymin=-Inf,ymax=Inf,fill=primary_cat),
            inherit.aes=FALSE, alpha=0.25) +
  geom_line(color="gray30", linewidth=0.3, alpha=0.8) +
  geom_vline(xintercept=with_tz(TRAIN_END,"America/New_York"),
             linetype="dashed", color="black") +
  scale_fill_manual(values=color_map, name="Storm type") +
  scale_y_continuous(labels=comma) +
  labs(title="Massachusetts Hourly Demand from Jan 2025 to Mar 2026",
       subtitle="Shading = NOAA storm windows Dashed = train/test split",
       x="Date (ET)", y="Demand (MWh)",
       caption="Source: EIA-930 + NOAA Storm Events") +
  theme_minimal(base_size=11) +
  theme(legend.position="top", axis.text.x=element_text(angle=30,hjust=1))

ggsave("/Users/ibabaig/Documents/school/ML/MLProject/plots/MA_01_demand_timeseries.png", p1, width=13, height=5, dpi=150)

# Plot 2 = Demand by storm category
p2 <- df %>%
  mutate(group=case_when(
    is_storm==0 ~ "Non-storm",
    str_detect(storm_category,"Snow") ~ "Snow storm",
    str_detect(storm_category,"Cold") ~ "Cold event",
    str_detect(storm_category,"Wind") ~ "Wind event",
    TRUE ~ "Other storm"
  )) %>%
  ggplot(aes(x=demand_mwh, fill=group)) +
  geom_density(alpha=0.45, adjust=1.2) +
  scale_x_continuous(labels=comma) +
  scale_fill_manual(values=c("Non-storm"="gray60","Snow storm"="steelblue",
                             "Cold event"="#2166ac","Wind event"="#d73027",
                             "Other storm"="gray40")) +
  labs(title="MA Demand Distribution by Extreme Weather Category",
       x="Demand (MWh)", y="Density", fill=NULL,
       caption="Source: EIA-930 + NOAA Storm Events") +
  theme_minimal(base_size=11) + theme(legend.position="top")

ggsave("/Users/ibabaig/Documents/school/ML/MLProject/plots/MA_02_demand_by_storm_type.png", p2, width=9, height=5, dpi=150)

# Plot 3 = Forecast vs actual
storm_in_test <- test %>% filter(is_storm==1)
if (nrow(storm_in_test) > 0) {
  first_storm <- as.character(storm_in_test$storm_label[1])
  s_start <- min(storm_in_test$datetime_et[storm_in_test$storm_label==first_storm])
  s_end   <- max(storm_in_test$datetime_et[storm_in_test$storm_label==first_storm])
  plot_df <- test %>%
    filter(datetime_et >= s_start - days(3),
           datetime_et <= s_end   + days(3))
  ptitle  <- paste("Bayesian Forecast Massachusetts ", first_storm)
} else {
  plot_df <- test %>% slice(1:336)
  ptitle  <- "Bayesian Forecast in Massachusetts (First Two Test Weeks)"
}

p_fc <- ggplot(plot_df, aes(x=datetime_et)) +
  geom_ribbon(aes(ymin=pred_lo90,ymax=pred_hi90), fill="steelblue", alpha=0.20) +
  geom_ribbon(aes(ymin=pred_lo50,ymax=pred_hi50), fill="steelblue", alpha=0.40) +
  geom_line(aes(y=pred_median), color="steelblue", linewidth=1.0) +
  geom_line(aes(y=demand_mwh),  color="black",     linewidth=0.6) +
  scale_y_continuous(labels=comma) +
  labs(title=ptitle,
       subtitle="Black = actual, Blue = posterior median, Bands = 50% and 90% CI",
       x="Date (ET)", y="MA Demand (MWh)",
       caption="Source: EIA-930 + NOAA Storm Events, Model: Bayesian Gibbs sampler") +
  theme_minimal(base_size=11)

ggsave("/Users/ibabaig/Documents/school/ML/MLProject/plots/MA_08_forecast_vs_actual.png", p_fc, width=12, height=5, dpi=150)

write_csv(df, "data/ma_model_ready.csv")

