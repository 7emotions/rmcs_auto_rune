#include "vc/feature/rune_tracker_ekf.h"

RuneTrackerEKF::RuneTrackerEKF()
    : m_x(cv::Matx<float, 6, 1>::zeros()),
      m_P(cv::Matx<float, 6, 6>::eye()),
      m_H(cv::Matx<float, 3, 6>::zeros()),
      m_initialized(false)
{
    // H = [I_3 | 0_3]: 仅观测位置分量
    m_H(0, 0) = 1.0f;
    m_H(1, 1) = 1.0f;
    m_H(2, 2) = 1.0f;
}

void RuneTrackerEKF::predict(double dt)
{
    if (!m_initialized)
        return;

    float fdt = static_cast<float>(dt);

    // 状态转移矩阵 F = [I_3 | dt*I_3; 0_3 | I_3]（恒速模型）
    cv::Matx<float, 6, 6> F = cv::Matx<float, 6, 6>::eye();
    F(0, 3) = fdt;
    F(1, 4) = fdt;
    F(2, 5) = fdt;

    // 过程噪声矩阵 Q = diag(qp, qp, qp, qv, qv, qv)
    float qp = rune_tracker_ekf_param.PROCESS_NOISE_POS;
    float qv = rune_tracker_ekf_param.PROCESS_NOISE_VEL;
    cv::Matx<float, 6, 6> Q = cv::Matx<float, 6, 6>::zeros();
    Q(0, 0) = qp; Q(1, 1) = qp; Q(2, 2) = qp;
    Q(3, 3) = qv; Q(4, 4) = qv; Q(5, 5) = qv;

    // 预测状态和协方差
    m_x = F * m_x;
    m_P = F * m_P * F.t() + Q;
}

void RuneTrackerEKF::update(const cv::Vec3f &measurement)
{
    if (!m_initialized)
    {
        // 用第一帧测量初始化状态（速度置零）
        m_x(0) = measurement[0];
        m_x(1) = measurement[1];
        m_x(2) = measurement[2];
        m_x(3) = 0.0f;
        m_x(4) = 0.0f;
        m_x(5) = 0.0f;

        // 初始协方差
        float r = rune_tracker_ekf_param.MEASUREMENT_NOISE;
        m_P = cv::Matx<float, 6, 6>::eye() * r;

        m_initialized = true;
        return;
    }

    // 测量噪声 R = r * I_3
    float r = rune_tracker_ekf_param.MEASUREMENT_NOISE;
    cv::Matx<float, 3, 3> R = cv::Matx<float, 3, 3>::eye() * r;

    // 新息: y = z - H * x
    cv::Matx<float, 3, 1> z(measurement[0], measurement[1], measurement[2]);
    cv::Matx<float, 3, 1> y = z - m_H * m_x;

    // 新息协方差: S = H * P * H^T + R
    cv::Matx<float, 3, 3> S = m_H * m_P * m_H.t() + R;

    // 卡尔曼增益: K = P * H^T * S^{-1}
    cv::Matx<float, 6, 3> K = m_P * m_H.t() * S.inv();

    // 更新状态: x = x + K * y
    m_x = m_x + K * y;

    // 更新协方差: P = (I - K * H) * P
    cv::Matx<float, 6, 6> I6 = cv::Matx<float, 6, 6>::eye();
    m_P = (I6 - K * m_H) * m_P;
}

cv::Point3f RuneTrackerEKF::getState() const
{
    return cv::Point3f(m_x(0), m_x(1), m_x(2));
}

cv::Point3f RuneTrackerEKF::getVelocity() const
{
    return cv::Point3f(m_x(3), m_x(4), m_x(5));
}

bool RuneTrackerEKF::isInitialized() const
{
    return m_initialized;
}

void RuneTrackerEKF::reset()
{
    m_x = cv::Matx<float, 6, 1>::zeros();
    m_P = cv::Matx<float, 6, 6>::eye();
    m_initialized = false;
}
