/**
 * @file rune_rotation_param.h
 * @brief 神符旋转参数模块头文件
 */

#pragma once
#include "vc/core/yml_manager.hpp"

//! 神符旋转参数模块
struct RuneRotationParam
{
    template <typename _Tp, typename Enable>
    friend void load(_Tp &, const std::string &);

    using paraId = void;

    //! 常量模式角速度 (rad/s)
    double CONST_W = 1.047;

    //! 正弦模式振幅 (rad/s)
    double SIN_A = 0.785;

    //! 正弦模式角频率 (rad/s)
    double SIN_B = 1.884;

    //! 正弦模式初相 (rad)
    double SIN_C = 0.0;

    //! 正弦模式直流偏置 (rad/s)
    double SIN_D = 1.305;

    YML_INIT(
        RuneRotationParam,
        YML_ADD_PARAM(CONST_W);
        YML_ADD_PARAM(SIN_A);
        YML_ADD_PARAM(SIN_B);
        YML_ADD_PARAM(SIN_C);
        YML_ADD_PARAM(SIN_D););
};

//! 神符旋转参数模块全局实例
inline RuneRotationParam rune_rotation_param;
