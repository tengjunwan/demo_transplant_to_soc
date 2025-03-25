#include "kalman_filter.h"


static KF_Mean mean = {
    .x = 0.0f,
    .y = 0.0f,
    .vx = 0.0f,
    .vy = 0.0f
};

static KF_Mean mean_bar = {
    .x = 0.0f,
    .y = 0.0f,
    .vx = 0.0f,
    .vy = 0.0f
};

static KF_CorvarianceMatrix sigma = {
    .var_x = 500.0f,
    .var_y = 500.0f,
    .var_vx = 500.0f,
    .var_vy = 500.0f,
    .cov_x_vx = 0.0f,
    .cov_y_vy = 0.0f
};

static KF_CorvarianceMatrix sigma_bar = {
    .var_x = 500.0f,
    .var_y = 500.0f,
    .var_vx = 500.0f,
    .var_vy = 500.0f,
    .cov_x_vx = 0.0f,
    .cov_y_vy = 0.0f
};


static float acc_std = 20.0f;
static float dt = 0.33333f;


void kf_init(float x, float y, float vx, float vy) {
    mean.x = x;
    mean.y = y;
    mean.vx = vx;
    mean.vy = vy;
    mean_bar.x = x;
    mean_bar.y = y;
    mean_bar.vx = vx;
    mean_bar.vy = vy;
    kf_correct(x, y, 0.1);
}


void kf_predict(float ux, float uy) {
    float m = acc_std * acc_std;
    float dt_power_2 = dt * dt;
    float dt_power_3 = dt * dt * dt;
    float dt_power_4 = dt_power_2 * dt_power_2;

    // update bel_bar distribution: mean
    mean_bar.x = mean.x + mean.vx * dt - 0.5 * dt_power_2 * ux;
    mean_bar.y = mean.y + mean.vy * dt - 0.5 * dt_power_2 * uy;
    mean_bar.vx = mean.vx - dt * ux;
    mean_bar.vy = mean.vy - dt * uy;

    // update bel_bar distribution: variance
    sigma_bar.var_x = sigma.var_x + 2 * dt * sigma.cov_x_vx + dt_power_2 * sigma.var_vx + 0.25 * dt_power_4 * m;
    sigma_bar.var_y = sigma.var_y + 2 * dt * sigma.cov_y_vy + dt_power_2 * sigma.var_vy + 0.25 * dt_power_4 * m;
    sigma_bar.var_vx = sigma.var_vx + dt_power_2 * m;
    sigma_bar.var_vy = sigma.var_vy + dt_power_2 * m;
    sigma_bar.cov_x_vx = sigma.cov_x_vx + dt * sigma.var_vx + 0.5 * dt_power_3 * m;
    sigma_bar.cov_y_vy = sigma.cov_y_vy + dt * sigma.var_vy + 0.5 * dt_power_3 * m;
}


void kf_correct(float zx, float zy, float meas_std) {
    float n = meas_std * meas_std;
    float term_1 = 1 / (sigma_bar.var_x + n);
    float term_2 = 1 / (sigma_bar.var_y + n);

    // update bel distribution: mean
    mean.x = mean_bar.x + sigma_bar.var_x * (zx - mean_bar.x) * term_1;
    mean.y = mean_bar.y + sigma_bar.var_y * (zy - mean_bar.y) * term_2;
    mean.vx = mean_bar.vx + sigma_bar.cov_x_vx * (zx - mean_bar.x) * term_1;
    mean.vy = mean_bar.vy + sigma_bar.cov_y_vy * (zy - mean_bar.y) * term_2;

    // update bel distribution: variance
    sigma.var_x = sigma_bar.var_x - sigma_bar.var_x * sigma_bar.var_x * term_1;
    sigma.var_y = sigma_bar.var_y - sigma_bar.var_y * sigma_bar.var_y * term_2;
    sigma.var_vx = sigma_bar.var_vx - sigma_bar.cov_x_vx * sigma_bar.cov_x_vx * term_1;
    sigma.var_vy = sigma_bar.var_vy - sigma_bar.cov_y_vy * sigma_bar.cov_y_vy * term_2;
    sigma.cov_x_vx = sigma_bar.cov_x_vx - sigma_bar.var_x * sigma_bar.cov_x_vx * term_1;
    sigma.cov_y_vy = sigma_bar.cov_y_vy - sigma_bar.var_y * sigma_bar.cov_y_vy * term_2;
}


void kf_correct_without_measurement(void) {
    mean.x = mean_bar.x;
    mean.y = mean_bar.y;
    mean.vx = mean_bar.vx;
    mean.vy = mean_bar.vy;
    sigma.var_x = sigma_bar.var_x;
    sigma.var_y = sigma_bar.var_y;
    sigma.var_vx = sigma_bar.var_vx;
    sigma.var_vy = sigma_bar.var_vy;
    sigma.cov_x_vx = sigma_bar.cov_x_vx;
    sigma.cov_y_vy = sigma_bar.cov_y_vy;
}


void kf_get_predicted_position(float* x, float* y) {
    *x = mean_bar.x;
    *y = mean_bar.y;
}


void kf_get_corrected_position(float* x, float* y) {
    *x = mean.x;
    *y = mean.y;
}





