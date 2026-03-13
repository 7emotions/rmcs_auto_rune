#pragma once

#include "rune_detect_demo/rune_detect_demo.h"
#include "vc/math/pose_node.hpp"
#include "vc/contour_proc/contour_wrapper.hpp"

#include "vc/feature/rune_target_inactive.h"
#include "vc/feature/rune_target_active.h"
#include "vc/feature/rune_fan_hump.h"
#include "vc/feature/rune_fan_active.h"
#include "vc/feature/rune_fan_inactive.h"
#include "vc/feature/rune_center.h"
#include "vc/feature/rune_group.h"
#include "vc/feature/rune_combo.h"
#include "vc/feature/rune_fan.h"

#include "vc/core/logging.h"
#include "vc/core/debug_tools/window_auto_layout.h"
#include "vc/core/debug_tools/param_view_manager.h"
#include "vc/core/yml_manager.hpp"
#include "vc/core/debug_tools.h"
#include "vc/core/command_line.h"

#include "vc/detector/rune_detector.h"


/**
 * @brief 核心处理函数
 * @param vid_cap 视频捕获对象
 */
void process(cv::VideoCapture &vid_cap);


/**
 * @brief 更新参数
 */
void updateParam(cv::VideoCapture &cap);

/**
 * @brief 命令行参数处理
 */
void parseCommandLine(int argc, char **argv);

/**
 * @brief 配置VideoCapture对象
 */
void setupVideoCapture(cv::VideoCapture &cap);

/**
 * @brief 计算旋转后的平移向量
 *
 * 通过 rune_group 获取神符中心的转轴（Z轴方向），根据角速度模式计算
 * theta，以转轴和 theta 构造轴角旋转，并将其应用到 tracker 的 tvec 上。
 *
 * @param[in] rune_group 神符序列组
 * @param[in] tracker 追踪器
 * @param[in] use_sine_mode 角速度模式选择 (false: 常量C, true: 正弦函数)
 * @param[in] dt_ms 时间步长（毫秒）
 * @return 旋转后的平移向量
 */
cv::Vec3d calcRotatedTvec(
    const std::shared_ptr<RuneGroup> &rune_group,
    const FeatureNode_cptr &tracker,
    bool use_sine_mode,
    double dt_ms);