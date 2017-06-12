#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // initialization flag
  is_initialized_ = false;

  // set initial time
  time_us_ = 0.0;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading paremeter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    if (i == 0) {
      weights_[i] = lambda_ / (lambda_ + n_aug_);
    } else {
      weights_[i] = 0.5 / (lambda_ + n_aug_);
    }
  }

  // NIS
  NIS_laser_ = 0.0;
  NIS_radar_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  // initialization
  if (!is_initialized_) {

    // initialize measurement
    x_ << 1, 1, 0, 0, 0;

    // initialize covariance matrix
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    time_us_ = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {

      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);      

    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {

      float ro = meas_package.raw_measurements_(0);
      float theta = meas_package.raw_measurements_(1);
      float ro_dot = meas_package.raw_measurements_(2);

      x_(0) = ro * cos(theta);
      x_(1) = ro * sin(theta);
      
    }

    is_initialized_ = true;
    return;
  }
  
  // calculate delta_t and set timestamp
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  // update state and state covariance matrix
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  /**
  * generate sigma points
  */
  // augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  // sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  /**
  * sigma points prediction
  */
  for (int i = 0; i < 2*n_aug_+1; i++) {
    float px       = Xsig_aug(0, i);
    float py       = Xsig_aug(1, i);
    float v        = Xsig_aug(2, i);
    float yaw      = Xsig_aug(3, i);
    float yawd     = Xsig_aug(4, i);
    float nu_a     = Xsig_aug(5, i);
    float nu_yawdd = Xsig_aug(6, i);
    
    // predict state values
    float px_p, py_p;
    if (fabs(yawd) <  1.0e-5) {
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    } else {
      px_p = px + v / yawd * (sin(yaw+yawd*delta_t) - sin(yaw));
      py_p = py + v / yawd * (-cos(yaw+yawd*delta_t) + cos(yaw));
    }

    float v_p = v;
    float yaw_p = yaw + yawd * delta_t;
    float yawd_p = yawd;

    // add noise
    px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p += nu_a * delta_t;
    yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p += nu_yawdd * delta_t;

    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  /**
  * predict mean and covariance
  */
  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // set measurement dimension
  int n_z = 2;

  // matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);

  // mean prediction
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix
  MatrixXd S = MatrixXd(n_z, n_z);

  // measurement
  VectorXd z = meas_package.raw_measurements_;


  for (int i = 0; i < 2* n_aug_+1; i++) {
    float px   = Xsig_pred_(0, i);
    float py   = Xsig_pred_(1, i);

    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  // calculate mean
  z_pred.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // calculate measurement covariance
  S.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {

    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;
  S += R;

  // cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  x_ += K * (z - z_pred);
  P_ += -K * S * K.transpose();

  VectorXd z_diff = z - z_pred;
  while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // set measurement dimension
  int n_z = 3;

  // matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
  Zsig.fill(0.0);

  // mean prediction
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix
  MatrixXd S = MatrixXd(n_z, n_z);

  // measurement
  VectorXd z = meas_package.raw_measurements_;

  for (int i = 0; i < 2* n_aug_+1; i++) {
    float px   = Xsig_pred_(0, i);
    float py   = Xsig_pred_(1, i);
    float v    = Xsig_pred_(2, i);
    float yaw  = Xsig_pred_(3, i);
    float yawd = Xsig_pred_(4, i);

    double c = px*px + py*py;
    if (fabs(c) < 1.0e-5) {
      c = 1.0e-5;
    }

    float ro = sqrt(c);
    float theta = atan2(py, px);
    float ro_dot = (px*cos(yaw)*v + py*sin(yaw)*v) / ro;

    Zsig(0, i) = ro;
    Zsig(1, i) = theta;
    Zsig(2, i) = ro_dot;
  }

  // calculate mean
  z_pred.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // calculate measurement covariance
  S.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S += R;

  // cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc * S.inverse();

  x_ += K * (z - z_pred);
  P_ += -K * S * K.transpose();

  VectorXd z_diff = z - z_pred;
  while (z_diff(1) >  M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
