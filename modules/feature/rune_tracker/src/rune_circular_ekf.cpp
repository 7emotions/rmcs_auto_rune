#include "vc/feature/rune_circular_ekf.h"
#include <cmath>

static constexpr float kTwoPi = 2.0f * static_cast<float>(M_PI);

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

float RuneCircularEKF::wrapAngle(float a)
{
    while (a >  static_cast<float>(M_PI)) a -= kTwoPi;
    while (a < -static_cast<float>(M_PI)) a += kTwoPi;
    return a;
}

// ---------------------------------------------------------------------------
// Constructor / mode control
// ---------------------------------------------------------------------------

RuneCircularEKF::RuneCircularEKF()
    : m_mode(Mode::CONSTANT),
      m_initialized(false),
      m_x_c(cv::Matx<float, 2, 1>::zeros()),
      m_P_c(cv::Matx<float, 2, 2>::eye()),
      m_x_s(cv::Matx<float, 4, 1>::zeros()),
      m_P_s(cv::Matx<float, 4, 4>::eye())
{}

void RuneCircularEKF::setMode(Mode mode)
{
    if (mode != m_mode)
    {
        m_mode = mode;
        reset();
    }
}

void RuneCircularEKF::reset()
{
    m_initialized = false;
    m_x_c = cv::Matx<float, 2, 1>::zeros();
    m_P_c = cv::Matx<float, 2, 2>::eye() * 1e2f;
    m_x_s = cv::Matx<float, 4, 1>::zeros();
    m_P_s = cv::Matx<float, 4, 4>::eye() * 1e2f;
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

void RuneCircularEKF::predict(double dt)
{
    if (!m_initialized)
        return;

    float fdt = static_cast<float>(dt);

    if (m_mode == Mode::CONSTANT)
    {
        // 线性恒速模型: [θ; ω] -> [θ + ω*dt; ω]
        cv::Matx<float, 2, 2> F(1.0f, fdt,
                                0.0f, 1.0f);

        cv::Matx<float, 2, 2> Q = cv::Matx<float, 2, 2>::zeros();
        Q(0, 0) = rune_circular_ekf_param.Q_THETA;
        Q(1, 1) = rune_circular_ekf_param.Q_OMEGA;

        m_x_c = F * m_x_c;
        m_P_c = F * m_P_c * F.t() + Q;
    }
    else // SINUSOIDAL
    {
        // 状态: [θ, A, phase, B]
        // 非线性预测:
        //   θ_new   = θ + (A·sin(phase) + B)·dt
        //   A_new   = A
        //   phase_new = phase + (2π/T)·dt
        //   B_new   = B
        const float theta = m_x_s(0);
        const float A     = m_x_s(1);
        const float phase = m_x_s(2);
        const float B     = m_x_s(3);

        const float omega_now = A * std::sin(phase) + B;
        const float dphase    = kTwoPi / rune_circular_ekf_param.SINUSOIDAL_PERIOD * fdt;

        cv::Matx<float, 4, 1> x_new;
        x_new(0) = theta + omega_now * fdt;
        x_new(1) = A;
        x_new(2) = phase + dphase;
        x_new(3) = B;
        m_x_s = x_new;

        // Jacobian ∂f/∂x（在预测前的状态处线性化）:
        // ∂θ_new/∂θ=1, ∂θ_new/∂A=sin(phase)*dt, ∂θ_new/∂phase=A·cos(phase)*dt, ∂θ_new/∂B=dt
        cv::Matx<float, 4, 4> F_jac = cv::Matx<float, 4, 4>::eye();
        F_jac(0, 1) = std::sin(phase) * fdt;
        F_jac(0, 2) = A * std::cos(phase) * fdt;
        F_jac(0, 3) = fdt;

        cv::Matx<float, 4, 4> Q = cv::Matx<float, 4, 4>::zeros();
        Q(0, 0) = rune_circular_ekf_param.Q_THETA;
        Q(1, 1) = rune_circular_ekf_param.Q_A;
        Q(2, 2) = rune_circular_ekf_param.Q_PHASE;
        Q(3, 3) = rune_circular_ekf_param.Q_B;

        m_P_s = F_jac * m_P_s * F_jac.t() + Q;
    }
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------

void RuneCircularEKF::update(float theta_meas)
{
    if (!m_initialized)
    {
        if (m_mode == Mode::CONSTANT)
        {
            m_x_c(0) = theta_meas;
            m_x_c(1) = rune_circular_ekf_param.INIT_OMEGA;
            m_P_c    = cv::Matx<float, 2, 2>::eye() * 1e2f;
        }
        else
        {
            m_x_s(0) = theta_meas;
            m_x_s(1) = rune_circular_ekf_param.INIT_A;
            m_x_s(2) = 0.0f; // 初始相位
            m_x_s(3) = rune_circular_ekf_param.INIT_B;
            m_P_s    = cv::Matx<float, 4, 4>::eye() * 1e2f;
        }
        m_initialized = true;
        return;
    }

    const float R = rune_circular_ekf_param.R_THETA;

    if (m_mode == Mode::CONSTANT)
    {
        // H = [1, 0]: 仅观测 θ
        // innovation（角度残差需要 wrap）
        const float y = wrapAngle(theta_meas - m_x_c(0));
        const float S = m_P_c(0, 0) + R;

        // K = P * H^T * S^{-1}，H^T = [1; 0]
        cv::Matx<float, 2, 1> K(m_P_c(0, 0) / S, m_P_c(1, 0) / S);

        m_x_c(0) = wrapAngle(m_x_c(0) + K(0) * y);
        m_x_c(1) += K(1) * y;

        // P = (I - K*H) * P
        const cv::Matx<float, 1, 2> H(1.0f, 0.0f);
        m_P_c = (cv::Matx<float, 2, 2>::eye() - K * H) * m_P_c;
    }
    else // SINUSOIDAL
    {
        // H = [1, 0, 0, 0]: 仅观测 θ
        const float y = wrapAngle(theta_meas - m_x_s(0));
        const float S = m_P_s(0, 0) + R;

        // K = P * H^T * S^{-1}，H^T = 第一列
        cv::Matx<float, 4, 1> K(
            m_P_s(0, 0) / S,
            m_P_s(1, 0) / S,
            m_P_s(2, 0) / S,
            m_P_s(3, 0) / S);

        m_x_s(0) = wrapAngle(m_x_s(0) + K(0) * y);
        m_x_s(1) += K(1) * y;
        m_x_s(2) += K(2) * y;
        m_x_s(3) += K(3) * y;

        const cv::Matx<float, 1, 4> H(1.0f, 0.0f, 0.0f, 0.0f);
        m_P_s = (cv::Matx<float, 4, 4>::eye() - K * H) * m_P_s;
    }
}

// ---------------------------------------------------------------------------
// predictAheadAngle
// ---------------------------------------------------------------------------

float RuneCircularEKF::predictAheadAngle(double dt) const
{
    if (!m_initialized)
        return 0.0f;

    float fdt = static_cast<float>(dt);

    if (m_mode == Mode::CONSTANT)
    {
        // θ + ω·dt
        return m_x_c(0) + m_x_c(1) * fdt;
    }
    else
    {
        // 精确积分: ∫₀ᵈᵗ (A·sin(phase₀ + 2π/T·s) + B) ds
        //  = -A·T/(2π) · (cos(phase₀ + 2π/T·dt) - cos(phase₀)) + B·dt
        const float theta = m_x_s(0);
        const float A     = m_x_s(1);
        const float phase = m_x_s(2);
        const float B     = m_x_s(3);

        const float omega_cycle = kTwoPi / rune_circular_ekf_param.SINUSOIDAL_PERIOD;
        float dtheta;
        if (std::abs(omega_cycle) > 1e-6f)
        {
            dtheta = -A / omega_cycle *
                     (std::cos(phase + omega_cycle * fdt) - std::cos(phase)) +
                     B * fdt;
        }
        else
        {
            // 退化情况（周期极大），用欧拉近似
            dtheta = (A * std::sin(phase) + B) * fdt;
        }
        return theta + dtheta;
    }
}

// ---------------------------------------------------------------------------
// predictAheadPos
// ---------------------------------------------------------------------------

cv::Point3f RuneCircularEKF::predictAheadPos(double dt, cv::Point3f center,
                                              float radius, float z_target) const
{
    if (!m_initialized)
        return cv::Point3f(0.0f, 0.0f, 0.0f);

    const float pred_theta = predictAheadAngle(dt);
    return cv::Point3f(
        center.x + radius * std::cos(pred_theta),
        center.y + radius * std::sin(pred_theta),
        z_target);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

float RuneCircularEKF::getTheta() const
{
    if (!m_initialized)
        return 0.0f;
    return (m_mode == Mode::CONSTANT) ? m_x_c(0) : m_x_s(0);
}

float RuneCircularEKF::getAngularVelocity() const
{
    if (!m_initialized)
        return 0.0f;
    if (m_mode == Mode::CONSTANT)
        return m_x_c(1);
    else
        return m_x_s(1) * std::sin(m_x_s(2)) + m_x_s(3); // A·sin(phase) + B
}
