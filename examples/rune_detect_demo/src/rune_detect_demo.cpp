#include "../include/rune_detect_demo/rune_detect_demo.h"
#include "../include/rune_detect_demo/rune_detect_demo_param.h"
#include "../include/rune_detect_demo/rune_rotation_param.h"
#include "vc/core/debug_tools/window_auto_layout.h"
#include "vc/dataio/dataio.h"
#include "vc/feature/rune_tracker.h"
#include "vc/math/pose_node.hpp"
#include "vc/math/transform6D.hpp"

#include <cmath>
#include <opencv2/core/utility.hpp>

#include <fstream>
#include <opencv2/core/utility.hpp>
#include <sstream>
using namespace cv;

void process(cv::VideoCapture& vid_cap) {
    static auto rune_groups = std::vector<FeatureNode_ptr>{};
    static auto rune_detector = RuneDetector::make_detector();
    static bool csv_initialized = false;
    static std::ofstream csv_file;

    // 读取识别参数
    // updateParam(vid_cap);

    Mat frame;
    vid_cap.read(frame);           // 从摄像头捕获一帧
    if (frame.empty()) {
        return;
    }
    DebugTools::get()->setImage(frame);
    DetectorInput input;
    DetectorOutput output;
    input.setImage(frame);
    input.setGyroData(GyroData()); // 空数据
    int64_t tick_ms = static_cast<int64_t>(cv::getTickCount() * 1000.0 / cv::getTickFrequency());
    input.setTick(cv::getTickCount());
    input.setColor(PixChannel::BLUE);
    input.setColorThresh(17);
    input.setFeatureNodes(rune_groups);
    rune_detector->detect(input, output);
    rune_groups = output.getFeatureNodes();
    if (rune_groups.empty())
        return;
    auto rune_group = RuneGroup::cast(rune_groups.front());
    if (rune_group->childFeatures().empty())
        return;
    // rune_group->setPredictFunc(
    //     [wg = std::weak_ptr<RuneGroup>(rune_group)](int64_t future_tick_ms) -> double {
    //         auto g = wg.lock();
    //         if (!g)
    //             return 0.0;

    //         const auto& ticks = g->getHistoryTicks();        // ms
    //         const auto& raw = g->getRawDatas();              // deg(roll)

    //         if (ticks.size() < 2 || raw.size() < 2) {
    //             return raw.empty() ? 0.0 : raw.front();
    //         }

    //         const double t0 = static_cast<double>(ticks[0]); // 最新
    //         const double t1 = static_cast<double>(ticks[1]); // 次新
    //         const double a0 = static_cast<double>(raw[0]);   // 最新 roll (deg)
    //         const double a1 = static_cast<double>(raw[1]);   // 次新 roll (deg)

    //         const double dt = (t0 - t1);                     // ms
    //         if (dt <= 1e-6)
    //             return a0;

    //         // deg/ms
    //         const double w = (a0 - a1) / dt;

    //         const double df = static_cast<double>(future_tick_ms) - t0; // ms
    //         return a0 + w * df;
    //     });

    FeatureNode_ptr target_tracker = nullptr;
    for (const auto& tracker : rune_group->getTrackers()) {
        auto tracker_ = TrackingFeatureNode::cast(tracker);
        if (tracker_->getHistoryNodes().size() < 2)
            continue;
        auto type = RuneCombo::cast(tracker_->getHistoryNodes().front())->getRuneType();
        if (type == RuneType::PENDING_STRUCK) {
            target_tracker = tracker;
            break;
        }
    }
    if (!target_tracker)
        return;

    auto center =
        RuneCombo::cast(TrackingFeatureNode::cast(target_tracker)->getHistoryNodes().front())
            ->getChildFeatures()
            .at(FeatureNode::ChildFeatureType::RUNE_CENTER);
    auto target =
        RuneCombo::cast(TrackingFeatureNode::cast(target_tracker)->getHistoryNodes().front())
            ->getChildFeatures()
            .at(FeatureNode::ChildFeatureType::RUNE_TARGET);
    auto rune_tracker = RuneTracker::cast(target_tracker);

    // 绘制
    Mat img_show = DebugTools::get()->getImage();
    rune_group->drawFeature(img_show);
}

cv::Vec3d calcRotatedTvec(
    const std::shared_ptr<RuneGroup>& rune_group, const FeatureNode_cptr& tracker,
    bool use_sine_mode, double dt_ms) {
    // 1. 通过rune_group获取神符中心的转轴
    PoseNode rune_to_cam;
    if (!rune_group->getCamPnpDataFromFilter(rune_to_cam))
        return Vec3d(0, 0, 0);

    // 转轴为神符坐标系的Z轴在相机坐标系下的方向
    Vec3d rotation_axis = rune_to_cam.rmat() * Vec3d(0, 0, 1);
    double axis_norm = cv::norm(rotation_axis);
    if (axis_norm < 1e-9)
        return Vec3d(0, 0, 0);
    rotation_axis /= axis_norm;

    // 2. 计算角速度w（两种模式）
    static double accumulated_time_s = 0.0;
    double dt_s = dt_ms / 1000.0;
    accumulated_time_s += dt_s;

    double w = 0.0;
    if (!use_sine_mode) {
        // 常量模式
        w = rune_rotation_param.CONST_W;
    } else {
        // 正弦模式: w = A * sin(B * t + C) + D
        w = rune_rotation_param.SIN_A
              * std::sin(rune_rotation_param.SIN_B * accumulated_time_s + rune_rotation_param.SIN_C)
          + rune_rotation_param.SIN_D;
    }

    // 3. 通过dt构造theta
    double theta = w * dt_s;

    // 4. 以转轴和theta构造轴角
    Vec3d axis_angle = rotation_axis * theta;

    // 5. 通过tracker获取pose的tvec
    auto tracker_ = TrackingFeatureNode::cast(tracker);
    if (!tracker_ || tracker_->getHistoryNodes().empty())
        return Vec3d(0, 0, 0);
    auto combo = tracker_->getHistoryNodes().front();
    if (!combo)
        return Vec3d(0, 0, 0);
    auto& pose_nodes = combo->getPoseCache().getPoseNodes();
    if (pose_nodes.find(CoordFrame::CAMERA) == pose_nodes.end())
        return Vec3d(0, 0, 0);
    Vec3d tvec = pose_nodes.at(CoordFrame::CAMERA).tvec();

    // 6. 将轴角的旋转应用到tvec并输出旋转后的tvec
    Matx33d R;
    cv::Rodrigues(axis_angle, R);
    Vec3d rotated_tvec = R * tvec;

    return rotated_tvec;
}

void updateParam(cv::VideoCapture& cap) {
    // 计数器
    static int count = 0;
    static std::string winname = "param";
    static int color_type_static = 0;
    static int color_thresh_static = 160;
    count++;

    // 判断窗口是否初始化
    auto layout = WindowAutoLayout::get();
    layout->addWindow(winname);

    // 判断参数滑动条是否存在
    static bool has_init_param = false;
    if (layout->hasWindow(winname) && !has_init_param) {
        //! 颜色类型参数类型
        std::string color_type_param_name = "Color Type (0: Red, 1: Blue)";
        //! 颜色二值化阈值
        std::string color_thresh_param_name = "Color Thresh (0~255)";
        createTrackbar(
            color_type_param_name, winname, nullptr, 1,
            [](int pos, void* userdata) {
                int* color_type_static_ptr = static_cast<int*>(userdata);
                *color_type_static_ptr = pos;
            },
            &color_type_static);
        setTrackbarPos(color_type_param_name, winname, 0);
        createTrackbar(
            color_thresh_param_name, winname, nullptr, 255,
            [](int pos, void* userdata) {
                int* color_thresh_static_ptr = static_cast<int*>(userdata);
                *color_thresh_static_ptr = pos;
            },
            &color_thresh_static);
        setTrackbarPos(color_thresh_param_name, winname, 100);
        has_init_param = true;
    }
    // 初始化进度条
    static bool has_init_progress = false;
    static int target_pos = 0;
    static int last_set_pos = -1; // 上次设置的位置
    do {
        if (rune_detect_demo_param.is_get_total_frames == false)
            break;
        if (layout->hasWindow(DebugTools::get()->getWindowName())) {
            if (!has_init_progress) {
                auto total_frames = rune_detect_demo_param.total_frames;
                createTrackbar(
                    "Frame Processed", DebugTools::get()->getWindowName(), nullptr, total_frames,
                    [](int pos, void* userdata) {
                        int* target_pos_ptr = static_cast<int*>(userdata);
                        *target_pos_ptr = pos;
                    },
                    &target_pos);
                setTrackbarPos("Frame Processed", DebugTools::get()->getWindowName(), 0);
                has_init_progress = true;
            }
            // 获取当前的进度条
            if (target_pos != last_set_pos) {
                // 设置视频帧位置
                cap.set(cv::CAP_PROP_POS_FRAMES, target_pos);
                last_set_pos = target_pos;
            }
            if (count % 50 == 0) // 每30帧更新一次
            {
                // 获取当前帧位置
                int current_frame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
                setTrackbarPos(
                    "Frame Processed", DebugTools::get()->getWindowName(), current_frame);
            }
        } else {
            has_init_progress = false;
        }

        // 判断是否到达视频末尾
        if (cap.get(cv::CAP_PROP_POS_FRAMES) >= cap.get(cv::CAP_PROP_FRAME_COUNT)) {
            // 重新开始
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            last_set_pos = -1;
            target_pos = 0;
            setTrackbarPos("Frame Processed", DebugTools::get()->getWindowName(), 0);
        }

    } while (0);

    // 进行参数赋值
    rune_detect_demo_param.color_type = color_type_static;
    rune_detect_demo_param.color_thresh = color_thresh_static;
}

void parseCommandLine(int argc, char** argv) {
    CommandLine cli;
    // 1. 帮助项说明 ：-h, --help
    cli.addOption("-h", "help", "显示帮助信息");
    // 2. 输入视频路径 ：-i, --input <path>
    cli.addOption("-i", "input", "输入视频路径", true);

    // 解析参数
    bool valid = cli.parse(argc, argv);
    if (!valid) {
        cli.printHelp(argv[0]);
        VC_WARNING_INFO(
            "程序使用示例：\n./VisCore_rune_detect_demo_exe -i "
            "./test_video/rune_video.mp4\n注意:\n 1.视频路径为绝对路径\n "
            "2.视频路径不要带外括号");
        exit(1);
    }

    if (cli.isSet("-h")) {
        cli.printHelp(argv[0]);
        exit(0);
    }

    if (cli.isSet("-i")) {
        rune_detect_demo_param.video_path = cli.get("-i");
        if (rune_detect_demo_param.video_path.empty()) {
            VC_WARNING_INFO("无效的视频路径");
        }
    }

    if (cli.getPositionalArgs().size() > 0) {
        VC_WARNING_INFO("检测到多余的参数，已忽略");
    }
}

void setupVideoCapture(cv::VideoCapture& cap) {
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (total_frames <= 0)
        VC_WARNING_INFO("无法获取视频总帧数，可能是摄像头或无效视频文件");
    else
        VC_PASS_INFO("视频总帧数: %d", total_frames);

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0)
        VC_WARNING_INFO("无法获取视频帧率，可能是摄像头或无效视频文件");
    else
        VC_PASS_INFO("视频帧率: %.2f", fps);

    bool is_get_total_frames = total_frames > 0;
    bool is_get_fps = fps > 0;
    if (!is_get_total_frames && !is_get_fps) {
        VC_WARNING_INFO("无法获取视频总帧数和帧率，默认按30FPS处理");
        cap.set(cv::CAP_PROP_FPS, 30);
    } else if (!is_get_total_frames && is_get_fps) {
        VC_PASS_INFO("只能获取视频帧率，按此帧率处理");
    } else if (is_get_total_frames && !is_get_fps) {
        VC_WARNING_INFO("只能获取视频总帧数，默认按30FPS处理");
        cap.set(cv::CAP_PROP_FPS, 30);
    } else {
        double duration = total_frames / fps;
        VC_PASS_INFO("视频时长: %.2f 秒", duration);
    }

    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 从头开始
    rune_detect_demo_param.is_get_fps = is_get_fps;
    rune_detect_demo_param.fps = fps;
    rune_detect_demo_param.is_get_total_frames = is_get_total_frames;
    rune_detect_demo_param.total_frames = total_frames;
}