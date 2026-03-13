/**
 * @file rune_tracker_ekf.h
 * @brief 神符追踪器扩展卡尔曼滤波器头文件
 * @author 张峰玮 (3480409161@qq.com)
 * @date 2025-08-24
 */

#pragma once
#include <opencv2/core.hpp>
#include "vc/core/yml_manager.hpp"

//! RuneTrackerEKF 参数模块
struct RuneTrackerEKFParam
{
    //! 位置过程噪声（越小输出越平滑，响应越慢）
    float PROCESS_NOISE_POS = 1e-2f;
    //! 速度过程噪声（越小速度估计越稳定）
    float PROCESS_NOISE_VEL = 1e-1f;
    //! 测量噪声（越大对测量值信任度越低，输出越平滑，抖动越小）
    float MEASUREMENT_NOISE = 50.0f;

    YML_INIT(
        RuneTrackerEKFParam,
        YML_ADD_PARAM(PROCESS_NOISE_POS);
        YML_ADD_PARAM(PROCESS_NOISE_VEL);
        YML_ADD_PARAM(MEASUREMENT_NOISE););
};

//! RuneTrackerEKF 参数实例
inline RuneTrackerEKFParam rune_tracker_ekf_param;

/**
 * @brief 神符追踪器 EKF 滤波器
 *
 * @note 使用恒速模型的扩展卡尔曼滤波器，对 3D 位置 (tvec) 进行滤波。
 *       状态向量: [x, y, z, vx, vy, vz]^T
 *       测量向量: [x, y, z]^T
 */
class RuneTrackerEKF
{
public:
    RuneTrackerEKF();
    ~RuneTrackerEKF() = default;

    /**
     * @brief 预测步骤（时间传播）
     *
     * @param[in] dt 时间步长（秒）
     */
    void predict(double dt);

    /**
     * @brief 更新步骤（测量更新）
     *
     * @param[in] measurement 测量值 [x, y, z]
     */
    void update(const cv::Vec3f &measurement);

    /**
     * @brief 获取滤波后的位置
     *
     * @return 滤波后的 3D 位置
     */
    cv::Point3f getState() const;

    /**
     * @brief 获取估计速度
     *
     * @return 估计的 3D 速度
     */
    cv::Point3f getVelocity() const;

    /**
     * @brief 滤波器是否已初始化
     *
     * @return true 已用第一帧测量初始化，false 尚未初始化
     */
    bool isInitialized() const;

    /**
     * @brief 重置滤波器状态
     */
    void reset();

private:
    //! 状态向量 [x, y, z, vx, vy, vz]^T
    cv::Matx<float, 6, 1> m_x;
    //! 状态协方差矩阵
    cv::Matx<float, 6, 6> m_P;
    //! 测量矩阵 H = [I_3 | 0_3]
    cv::Matx<float, 3, 6> m_H;
    //! 是否已用第一帧测量初始化
    bool m_initialized;
};
