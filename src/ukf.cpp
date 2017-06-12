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

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
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
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {

    // initialize measurement
    x_ << 1, 1, 1, 1, 0.1;

    // initialize covariance matrix
    P_ << 0.15,    0, 0, 0, 0,
             0, 0.15, 0, 0, 0,
             0,    0, 1, 0, 0,
             0,    0, 0, 1, 0,
             0,    0, 0, 0, 1;

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
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /**
  * generate sigma points
  */
  // augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  cout << x_aug << endl;

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
    if (fabs(yawd) <  1.0e-10) {
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
    P_ += weights_(i) * (Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i) - x_).transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
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
    S += weights_(i) * (Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose();
  }
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;
  S += R;

  // cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  }

  MatrixXd K = Tc * S.inverse();

  x_ += K * (z - z_pred);
  P_ += -K * S * K.transpose();

  NIS_laser_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
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

    float ro = sqrt(px*px + py*py);
    float theta = atan(py/px);
    float ro_dot;
    if (ro < 1.0e-5) {
      ro_dot = (px*cos(yaw)*v + py*sin(yaw)*v) / 1.0e-5;
    } else {
      ro_dot = (px*cos(yaw)*v + py*sin(yaw)*v) / ro;
    }

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
    S += weights_(i) * (Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose();
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
    Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  }

  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc * S.inverse();

  x_ += K * (z - z_pred);
  P_ += -K * S * K.transpose();

  NIS_radar_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
}
