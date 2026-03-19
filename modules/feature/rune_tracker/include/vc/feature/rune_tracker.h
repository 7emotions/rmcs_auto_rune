/**
 * @file rune_tracker.h
 * @brief 神符时间序列追踪器头文件
 * @author 张峰玮 (3480409161@qq.com)
 * @date 2025-08-24
 */

#pragma once
#include "vc/feature/tracking_feature_node.h"
#include "vc/feature/rune_tracker_ekf.h"
#include "vc/feature/rune_circular_ekf.h"

//! 神符时间序列追踪器
class RuneTracker : public TrackingFeatureNode
{
    using Ptr = std::shared_ptr<RuneTracker>;

    //! 掉帧数
    DEFINE_PROPERTY_WITH_INIT(DropFrameCount, public, protected, (int), 0);

public:
    RuneTracker() = default;
    RuneTracker(const RuneTracker &) = delete;
    RuneTracker(RuneTracker &&) = delete;
    virtual ~RuneTracker() = default;

    /**
     * @brief 构建 RuneTracker
     *
     * @return 新建的 RuneTracker 智能指针
     */
    static Ptr make_feature() { return std::make_shared<RuneTracker>(); };

    /**
     * @brief 动态类型转换
     *
     * @param[in] p_tracker 抽象 FeatureNode 指针
     * @return 转换后的 RuneTracker 智能指针
     */
    static inline Ptr cast(FeatureNode_ptr p_tracker)
    {
        return std::dynamic_pointer_cast<RuneTracker>(p_tracker);
    }

    /**
     * @brief 更新时间序列
     *
     * @param[in] p_rune 神符共享指针
     * @param[in] tick 当前时间戳
     * @param[in] gyro_data 云台陀螺仪数据
     */
    void update(FeatureNode_ptr p_rune, int64 tick, const GyroData &gyro_data);

    /**
     * @brief 更新可见性
     *
     * @param[in] is_visible 是否可见
     */
    void updateVisible(bool is_visible);

    /**
     * @brief 绘制特征节点
     *
     * @param[in,out] image 绘制目标图像
     * @param[in] config 绘制配置指针，默认为nullptr
     *
     * @note 默认实现为空，子类可以重写此函数以实现具体绘制逻辑。
     */
    virtual void drawFeature(cv::Mat &image, const DrawConfig_cptr &config = nullptr) const override;

    /**
     * @brief 获取中心位置的 EKF 滤波结果
     *
     * @return 滤波后的中心 3D 位置（相机坐标系）
     */
    cv::Point3f getFilteredCenterPos() const { return m_center_ekf.getState(); }

    /**
     * @brief 获取靶心位置的 EKF 滤波结果
     *
     * @return 滤波后的靶心 3D 位置（相机坐标系）
     */
    cv::Point3f getFilteredTargetPos() const { return m_target_ekf.getState(); }

    /**
     * @brief 预测 dt 秒后的神符中心位置（不修改滤波器状态）
     *
     * @param[in] dt_sec 向前预测的时间（秒）
     * @return 预测的 3D 中心位置（相机坐标系），基于恒速模型: pos + vel * dt
     */
    cv::Point3f getPredictedCenterPos(double dt_sec) const { return m_center_ekf.predictAhead(dt_sec); }

    /**
     * @brief 预测 dt 秒后的神符靶心位置（不修改滤波器状态）
     *
     * 若圆周运动 EKF 已初始化，则使用当前运动模式（恒角速度或正弦角速度）约束下的
     * 圆周预测；否则退化为 Cartesian 恒速模型预测。
     *
     * @param[in] dt_sec 向前预测的时间（秒）
     * @return 预测的 3D 靶心位置（相机坐标系）
     */
    cv::Point3f getPredictedTargetPos(double dt_sec) const;

    /**
     * @brief 设置靶心运动模式（切换后圆周 EKF 自动重置）
     *
     * @param[in] is_sinusoidal true → 正弦角速度大符模式，false → 恒角速度小符模式
     */
    void setMotionMode(bool is_sinusoidal);

    //! 当前运动模式是否为正弦模式
    bool isMotionSinusoidal() const
    {
        return m_circular_ekf.getMode() == RuneCircularEKF::Mode::SINUSOIDAL;
    }

    //! 圆周运动 EKF 是否已初始化
    bool isCircularEKFInitialized() const { return m_circular_ekf.isInitialized(); }

    //! 当前估计角速度（rad/s）
    float getAngularVelocity() const { return m_circular_ekf.getAngularVelocity(); }

    bool isPredictionStable() const;

    void markPredictionUnstable(int cooldown_frames = -1);

    /**
     * @brief 中心 EKF 是否已用首帧测量初始化
     */
    bool isCenterEKFInitialized() const { return m_center_ekf.isInitialized(); }

    /**
     * @brief 靶心 EKF 是否已用首帧测量初始化
     */
    bool isTargetEKFInitialized() const { return m_target_ekf.isInitialized(); }

private:
    void advancePredictionStability();
    void resetTrackingState();

    /**
     * @brief 从神符组合体更新内部数据
     *
     * @param[in] p_combo 神符组合体共享指针
     */
    void updateFromRune(FeatureNode_ptr p_combo);

    //! 神符中心位置 EKF 滤波器
    RuneTrackerEKF m_center_ekf;
    //! 神符靶心位置 EKF 滤波器（Cartesian 恒速模型，用于位置平滑）
    RuneTrackerEKF m_target_ekf;
    //! 神符靶心圆周运动 EKF（角度状态，用于运动约束预测）
    RuneCircularEKF m_circular_ekf;
    //! 靶心到中心的当前半径（XY 平面内）
    float m_current_radius = 0.0f;
    //! 靶心当前 Z 坐标（用于 3D 还原）
    float m_current_target_z = 0.0f;
    //! 上一帧时间戳（用于计算 dt）
    int64_t m_prev_tick = 0;
    int m_prediction_stable_frames = 0;
    int m_prediction_unstable_cooldown = 0;
};

//! 神符追踪器智能指针类型
using RuneTracker_ptr = std::shared_ptr<RuneTracker>;
using RuneTracker_cptr = std::shared_ptr<const RuneTracker>;
