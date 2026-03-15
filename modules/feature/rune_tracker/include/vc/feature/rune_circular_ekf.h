/**
 * @file rune_circular_ekf.h
 * @brief 神符靶心圆周运动 EKF 头文件
 * @author 张峰玮 (3480409161@qq.com)
 * @date 2025-08-24
 */

#pragma once
#include <opencv2/core.hpp>
#include "vc/core/yml_manager.hpp"

//! RuneCircularEKF 参数模块
struct RuneCircularEKFParam
{
    // -------- 公共参数 --------
    //! 角度过程噪声
    float Q_THETA = 1e-4f;
    //! 角度测量噪声（atan2 测量值的方差，单位 rad²）
    float R_THETA = 1e-2f;

    // -------- 恒角速度模式 (CONSTANT) --------
    //! 角速度过程噪声
    float Q_OMEGA = 1e-3f;
    //! 初始角速度猜测值（rad/s，正为逆时针）
    float INIT_OMEGA = 1.5f;

    // -------- 正弦角速度模式 (SINUSOIDAL) --------
    //! ω(t) = A·sin(2π/T·t + phase) + B
    //! 正弦周期 T（秒），对应 RoboMaster 大符标准值 2π/1.884 ≈ 3.33 s
    float SINUSOIDAL_PERIOD = 3.33f;
    //! 振幅过程噪声
    float Q_A = 1e-3f;
    //! 相位过程噪声（相位由运动方程确定性推进，此噪声补偿模型误差）
    float Q_PHASE = 1e-6f;
    //! 直流偏置过程噪声
    float Q_B = 1e-3f;
    //! 振幅初始猜测值（rad/s）
    float INIT_A = 0.785f;
    //! 直流偏置初始猜测值（rad/s）
    float INIT_B = 1.305f;

    YML_INIT(
        RuneCircularEKFParam,
        YML_ADD_PARAM(Q_THETA);
        YML_ADD_PARAM(R_THETA);
        YML_ADD_PARAM(Q_OMEGA);
        YML_ADD_PARAM(INIT_OMEGA);
        YML_ADD_PARAM(SINUSOIDAL_PERIOD);
        YML_ADD_PARAM(Q_A);
        YML_ADD_PARAM(Q_PHASE);
        YML_ADD_PARAM(Q_B);
        YML_ADD_PARAM(INIT_A);
        YML_ADD_PARAM(INIT_B););
};

//! RuneCircularEKF 参数实例
inline RuneCircularEKFParam rune_circular_ekf_param;

/**
 * @brief 神符靶心圆周运动 EKF
 *
 * 将靶心绕中心的旋转约束接入追踪器：两种运动模式均以角度 θ 作为状态，
 * 通过 atan2(target - center) 作为观测量进行 EKF 更新。
 *
 * **CONSTANT 模式（恒角速度）**
 * - 状态向量: [θ, ω]
 * - 预测: θ += ω·dt，ω 恒定（线性 Kalman 滤波器）
 *
 * **SINUSOIDAL 模式（正弦角速度）**
 * - 运动方程: ω(t) = A·sin(phase) + B，phase 以 2π/T 确定性推进
 * - 状态向量: [θ, A, phase, B]
 * - 预测: 非线性 EKF，Jacobian 解析计算
 *
 * @note predictAheadAngle() 对正弦模式采用精确积分而非欧拉近似，
 *       从而给出无累积误差的向前预测角度。
 */
class RuneCircularEKF
{
public:
    //! 运动模式
    enum class Mode
    {
        CONSTANT,   //!< 恒角速度
        SINUSOIDAL  //!< 正弦角速度（大符模式）
    };

    RuneCircularEKF();
    ~RuneCircularEKF() = default;

    /**
     * @brief 切换运动模式
     *
     * 切换后自动 reset() 以清除旧状态。
     * @param mode 目标模式
     */
    void setMode(Mode mode);

    //! 当前运动模式
    Mode getMode() const { return m_mode; }

    /**
     * @brief 预测步骤（时间传播，修改内部状态）
     * @param dt 时间步长（秒）
     */
    void predict(double dt);

    /**
     * @brief 更新步骤（角度测量融合）
     * @param theta_meas 原始角度测量值（rad），由 atan2(target-center) 计算
     */
    void update(float theta_meas);

    /**
     * @brief 向前预测 dt 秒后的旋转角度（不修改内部状态）
     *
     * - CONSTANT:   θ + ω·dt
     * - SINUSOIDAL: 对 ω(t) 精确积分，θ - A·T/(2π)·(cos(phase+2π/T·dt) - cos(phase)) + B·dt
     *
     * @param dt 预测时间（秒）
     * @return 预测角度（rad）
     */
    float predictAheadAngle(double dt) const;

    /**
     * @brief 向前预测 dt 秒后的靶心 3D 位置（不修改内部状态）
     *
     * 利用 center 和 radius 将预测角度还原为相机坐标系下的 3D 位置。
     *
     * @param dt         预测时间（秒）
     * @param center     神符中心 3D 位置（相机坐标系）
     * @param radius     靶心到中心的当前距离（XY 平面内）
     * @param z_target   靶心当前 Z 坐标（保留 Z 分量）
     * @return 预测的靶心 3D 位置
     */
    cv::Point3f predictAheadPos(double dt, cv::Point3f center,
                                float radius, float z_target) const;

    //! 滤波器是否已初始化
    bool isInitialized() const { return m_initialized; }

    //! 重置滤波器状态
    void reset();

    //! 当前估计角度（rad）
    float getTheta() const;

    //! 当前估计角速度（rad/s）
    float getAngularVelocity() const;

private:
    //! 角度规范化到 [-π, π]
    static float wrapAngle(float a);

    Mode m_mode;
    bool m_initialized;

    // ---- CONSTANT 模式: 状态 [θ, ω] ----
    cv::Matx<float, 2, 1> m_x_c;  //!< 状态向量
    cv::Matx<float, 2, 2> m_P_c;  //!< 协方差矩阵

    // ---- SINUSOIDAL 模式: 状态 [θ, A, phase, B] ----
    cv::Matx<float, 4, 1> m_x_s;  //!< 状态向量
    cv::Matx<float, 4, 4> m_P_s;  //!< 协方差矩阵
};
