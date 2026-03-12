#include "include/rune_detect_demo/rune_detect_demo.h"
#include "include/rune_detect_demo/rune_detect_demo_param.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    parseCommandLine(argc, argv);
    VC_PASS_INFO("Video Path: %s", rune_detect_demo_param.video_path.c_str());
    VideoCapture cap(rune_detect_demo_param.video_path);
    if (!cap.isOpened())
        VC_THROW_ERROR("无法打开视频文件: %s", rune_detect_demo_param.video_path.c_str());
    setupVideoCapture(cap);
    static bool has_init_progress = false;
    static int target_pos = 0;
    static int last_set_pos = -1; // 上次设置的位置
    while (true) {
        process(cap);
        DebugTools::get()->show();
        waitKey(1);
        if (cap.get(cv::CAP_PROP_POS_FRAMES) >= cap.get(cv::CAP_PROP_FRAME_COUNT)) {
            // 重新开始
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            last_set_pos = -1;
            target_pos = 0;
            setTrackbarPos("Frame Processed", DebugTools::get()->getWindowName(), 0);
        }
    }
}
