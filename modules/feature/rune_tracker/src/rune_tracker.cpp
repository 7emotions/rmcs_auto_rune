#include "vc/feature/rune_tracker.h"
#include "vc/feature/rune_tracker_param.h"
#include "vc/feature/rune_combo.h"
#include "vc/camera/camera_param.h"
#include "vc/math/pose_node.hpp"
#include <cmath>

using namespace std;
using namespace cv;

void RuneTracker::updateFromRune(FeatureNode_ptr p_combo)
{
    setImageCache(p_combo->getImageCache());
    setPoseCache(p_combo->getPoseCache());
    getHistoryNodes().push_front(p_combo);
    getHistoryTicks().push_front(p_combo->getTick());

    int64_t current_tick = p_combo->getTick();

    // 计算 dt 并执行预测步骤
    if (m_prev_tick > 0 && (m_center_ekf.isInitialized() || m_target_ekf.isInitialized()))
    {
        double dt = static_cast<double>(current_tick - m_prev_tick) / cv::getTickFrequency();
        // 合理性校验：dt 超出 [0, 1) 秒时跳过预测（防止首帧或长时间掉帧后跳变）
        if (dt > 0.0 && dt < 1.0)
        {
            m_center_ekf.predict(dt);
            m_target_ekf.predict(dt);
            m_circular_ekf.predict(dt);
        }
    }
    m_prev_tick = current_tick;

    // 用子特征的 PnP tvec 更新 EKF
    const auto &child_features = p_combo->getChildFeatures();

    if (child_features.count(FeatureNode::ChildFeatureType::RUNE_CENTER) > 0)
    {
        const auto &center = child_features.at(FeatureNode::ChildFeatureType::RUNE_CENTER);
        const auto &pose_nodes = center->getPoseCache().getPoseNodes();
        if (pose_nodes.count(CoordFrame::CAMERA) > 0)
        {
            const auto &tvec = pose_nodes.at(CoordFrame::CAMERA).tvec();
            m_center_ekf.update(Vec3f(
                static_cast<float>(tvec[0]),
                static_cast<float>(tvec[1]),
                static_cast<float>(tvec[2])));
        }
    }

    if (child_features.count(FeatureNode::ChildFeatureType::RUNE_TARGET) > 0)
    {
        const auto &target = child_features.at(FeatureNode::ChildFeatureType::RUNE_TARGET);
        const auto &pose_nodes = target->getPoseCache().getPoseNodes();
        if (pose_nodes.count(CoordFrame::CAMERA) > 0)
        {
            const auto &tvec = pose_nodes.at(CoordFrame::CAMERA).tvec();
            m_target_ekf.update(Vec3f(
                static_cast<float>(tvec[0]),
                static_cast<float>(tvec[1]),
                static_cast<float>(tvec[2])));
        }
    }

    // 使用滤波后的 center 和 target 位置更新圆周运动 EKF
    if (m_center_ekf.isInitialized() && m_target_ekf.isInitialized())
    {
        const cv::Point3f fc = m_center_ekf.getState();
        const cv::Point3f ft = m_target_ekf.getState();

        const float dx = ft.x - fc.x;
        const float dy = ft.y - fc.y;
        m_current_radius  = std::sqrt(dx * dx + dy * dy);
        m_current_target_z = ft.z;

        if (m_current_radius > 1e-3f)
        {
            const float theta_meas = std::atan2(dy, dx);
            m_circular_ekf.update(theta_meas);
        }
    }
}

void RuneTracker::update(FeatureNode_ptr p_rune, int64 tick, const GyroData &gyro_data)
{
    // 数据更新
    updateFromRune(p_rune);
    auto &__history_deque = getHistoryNodes();
    auto &__tick_deque = getHistoryTicks();

    if (__history_deque.size() >= static_cast<size_t>(rune_tracker_param.MAX_DEQUE_SIZE))
        __history_deque.pop_back();
    if (__tick_deque.size() >= static_cast<size_t>(rune_tracker_param.MAX_DEQUE_SIZE))
        __tick_deque.pop_back();
}

void RuneTracker::updateVisible(bool is_visible)
{
    auto &__vanish_num = getDropFrameCount();
    if (is_visible)
    {
        __vanish_num = 0;
    }
    else
    {
        ++__vanish_num;
    }
}

cv::Point3f RuneTracker::getPredictedTargetPos(double dt_sec) const
{
    // 优先使用圆周运动 EKF（具有正确的运动模型约束）
    if (m_circular_ekf.isInitialized() && m_current_radius > 1e-3f)
    {
        // 中心位置也向前预测，以补偿相机运动
        const cv::Point3f predicted_center = m_center_ekf.predictAhead(dt_sec);
        return m_circular_ekf.predictAheadPos(dt_sec, predicted_center,
                                               m_current_radius, m_current_target_z);
    }
    // 退化：圆周 EKF 尚未初始化时，回退到 Cartesian 恒速预测
    return m_target_ekf.predictAhead(dt_sec);
}

void RuneTracker::setMotionMode(bool is_sinusoidal)
{
    m_circular_ekf.setMode(is_sinusoidal ? RuneCircularEKF::Mode::SINUSOIDAL
                                         : RuneCircularEKF::Mode::CONSTANT);
}

inline void drawPentagonWedge(cv::Mat &img, const PoseNode &p,
                              float radius, float height,
                              int sector_index = 0,
                              cv::Point2f center_offset = cv::Point2f(0.0f, 0.0f),
                              cv::Scalar c = cv::Scalar(0, 255, 0))
{
    if (radius <= 0.0f || height == 0.0f)
        return;

    float hz = height / 2.0f; // 上下各半高
    const float angle_step = 2.0f * CV_PI / 5.0f;

    // 规范化 sector_index
    int i0 = ((sector_index % 5) + 5) % 5;

    // 当前扇区的两个顶点角度（保持右手系）
    float angle1 = -(i0 * angle_step - CV_PI / 2.0f);
    float angle2 = -(((i0 + 1) % 5) * angle_step - CV_PI / 2.0f);

    // --- 构造 3D 顶点 ---
    std::vector<cv::Point3f> pts3d;
    // 下底（三角形 z = -hz）
    pts3d.emplace_back(center_offset.x, center_offset.y, -hz); // 0: 底中心
    pts3d.emplace_back(center_offset.x + radius * std::cos(angle1),
                       center_offset.y + radius * std::sin(angle1), -hz); // 1: 底顶点1
    pts3d.emplace_back(center_offset.x + radius * std::cos(angle2),
                       center_offset.y + radius * std::sin(angle2), -hz); // 2: 底顶点2
    // 上底（三角形 z = +hz）
    pts3d.emplace_back(center_offset.x, center_offset.y, hz); // 3: 顶中心
    pts3d.emplace_back(center_offset.x + radius * std::cos(angle1),
                       center_offset.y + radius * std::sin(angle1), hz); // 4: 顶顶点1
    pts3d.emplace_back(center_offset.x + radius * std::cos(angle2),
                       center_offset.y + radius * std::sin(angle2), hz); // 5: 顶顶点2

    // --- 投影到图像平面 ---
    std::vector<cv::Point2f> pts2d;
    projectPoints(pts3d, p.rvec(), p.tvec(),
                  camera_param.cameraMatrix, camera_param.distCoeff, pts2d);

    // --- 绘制线条 ---
    // 下底
    cv::line(img, pts2d[0], pts2d[1], c, 1);
    cv::line(img, pts2d[1], pts2d[2], c, 1);
    cv::line(img, pts2d[2], pts2d[0], c, 1);

    // 上底
    cv::line(img, pts2d[3], pts2d[4], c, 1);
    cv::line(img, pts2d[4], pts2d[5], c, 1);
    cv::line(img, pts2d[5], pts2d[3], c, 1);

    // 竖直连线
    cv::line(img, pts2d[0], pts2d[3], c, 1);
    cv::line(img, pts2d[1], pts2d[4], c, 1);
    cv::line(img, pts2d[2], pts2d[5], c, 1);
}

inline void drawCube(cv::Mat &img, const PoseNode &p, float x_len, float y_len, float z_len, cv::Scalar c)
{
    float hx = x_len / 2, hy = y_len / 2, hz = z_len / 2;
    vector<Point3f> pts3d = {{-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz}, {-hx, -hy, hz}, {hx, -hy, hz}, {hx, hy, hz}, {-hx, hy, hz}};
    vector<Point2f> pts2d;
    projectPoints(pts3d, p.rvec(), p.tvec(), camera_param.cameraMatrix, camera_param.distCoeff, pts2d);
    for (int i = 0; i < 4; i++)
    {
        line(img, pts2d[i], pts2d[(i + 1) % 4], c, 1);
        line(img, pts2d[i + 4], pts2d[(i + 1) % 4 + 4], c, 1);
        line(img, pts2d[i], pts2d[i + 4], c, 1);
    }
}

void RuneTracker::drawFeature(cv::Mat &image, const DrawConfig_cptr &config) const
{
    auto &pose_info = getPoseCache();
    do
    {
        if (pose_info.getPoseNodes().count(CoordFrame::CAMERA) == 0)
            break;
        auto &p = pose_info.getPoseNodes().at(CoordFrame::CAMERA);
        Scalar color = Scalar(0, 255, 0);
        auto type = RuneCombo::cast(this->getHistoryNodes().front())->getRuneType();
        if(type == RuneType::STRUCK)
            color = Scalar(0, 255, 0);
        else if(type == RuneType::PENDING_STRUCK)
            color = Scalar(0, 255, 255);
        else if(type == RuneType::UNKNOWN || type == RuneType::UNSTRUCK)
            color = Scalar(255, 255, 255);
        drawCube(image, p, 500, 500, 300, color);
    } while (0);
}
