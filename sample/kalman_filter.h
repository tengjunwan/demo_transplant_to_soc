#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
} KF_Mean;


typedef struct {
    float var_x;
    float var_y;
    float var_vx;
    float var_vy;
    float cov_x_vx;
    float cov_y_vy;
} KF_CorvarianceMatrix;



void kf_init(float x, float y, float vx, float vy);
void kf_predict(float ux, float uy);
void kf_correct(float zx, float zy, float meas_std);
void kf_correct_without_measurement(void);
void kf_get_predicted_position(float* x, float* y);
void kf_get_corrected_position(float* x, float* y);
void kf_set_velocity(float vx, float vy);


#endif