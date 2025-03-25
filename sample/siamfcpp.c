#include <stdio.h>

#include "sample_comm.h"
#include "sample_common_ive.h"
#include "sample_npu_process.h"

#include "siamfcpp_process.h"
#include "siamfcpp.h"
#include "createFrame.h"
#include "frameProcess.h"
#include "kalman_filter.h"


#define TEMPLATE_INPUT_SIZE 127
#define TEMPLATE_FAKE_INPUT_SIZE 128
#define SEARCH_INPUT_SIZE 303
#define SEARCH_FAKE_INPUT_SIZE 304
#define SCORE_THRESHOLD 0.0f
#define USE_KALMAN_FILTER 0

static ot_vpss_grp g_vpssGrp;
static ot_vpss_chn g_vpssChn;
static int g_thdThread = 0;
static stmTrackerState *stateInfo;
static pthread_mutex_t algolock = PTHREAD_MUTEX_INITIALIZER;

static long getms() {
    struct timeval start;
    gettimeofday(&start, NULL);
    long ms = (start.tv_sec) * 1000 + (start.tv_usec) / 1000;
    return ms;
}

static int next_multiple_of_16(int x) {
    return ((x / 16) + 1) * 16;
}

static void siamfcppTemplateGetCrop(stmTrackerState* state, float context_amount, int crop[4]) {
    float w = state->w;
    float h = state->h;
    float wc = w + context_amount * (w + h);
    float hc = h + context_amount * (w + h);
    float size_template_crop = sqrtf(wc * hc);

    // ensure size is a multiple of 16
    size_template_crop = (float)next_multiple_of_16((int)size_template_crop);

    // update state scale
    state->scale = size_template_crop / TEMPLATE_INPUT_SIZE;

    // calculate crop rectangle
    crop[0] = (int)(state->cx - size_template_crop * 0.5f);  // crop x
    crop[1] = (int)(state->cy - size_template_crop * 0.5f);  // crop y
    crop[2] = (int)size_template_crop;  // crop w
    crop[3] = (int)size_template_crop;  // crop h

    // force crop x to be even since crop on YUV420SP
    crop[0] &= ~1;
}


static void siamfcppSearchGetCrop(stmTrackerState* state, float context_amount, int crop[4]) {
    float w = state->w;
    float h = state->h;
    float wc = w + context_amount * (w + h);
    float hc = h + context_amount * (w + h);
    float size_template_crop = sqrtf(wc * hc);
    state->scale = size_template_crop / TEMPLATE_INPUT_SIZE;
    float size_search_crop = SEARCH_INPUT_SIZE * state->scale;

    // ensure size is a multiple of 16
    size_search_crop = (float)next_multiple_of_16((int)size_search_crop);

    // calculate crop rectangle
    crop[0] = (int)(state->cx - size_search_crop * 0.5f);  // crop x
    crop[1] = (int)(state->cy - size_search_crop * 0.5f);  // crop y
    crop[2] = (int)size_search_crop;  // crop w
    crop[3] = (int)size_search_crop;  // crop h

    // force crop x to be even since crop on YUV420SP
    crop[0] &= ~1;
}





void siamfcpp_proc_init(int32_t vpss_grp, int32_t vpss_chn) {
    g_vpssGrp = vpss_grp;
    g_vpssChn = vpss_chn;
    g_thdThread = 1;
    stateInfo = (stmTrackerState*)malloc(sizeof(stmTrackerState));
}


void siamfcpp_proc_uninit(void) {
    g_thdThread = 0;
}


void *siamfcpp_proc_run(void *parg) {
    td_s32 ret;
    ot_video_frame_info frame_info = {0};
    const ot_vpss_grp vpss_grp = g_vpssGrp;
    const ot_vpss_chn vpss_chn = g_vpssChn + 1;
    const td_s32 milli_sec = 200;

    int policy;
    struct sched_param param;
    pthread_getschedparam(pthread_self(), &policy, &param);
    policy = SCHED_FIFO;
    param.sched_priority = sched_get_priority_max(policy);
    pthread_setschedparam(pthread_self(), policy, &param);

    siamfcpp_init();

    stmTrackerState *state = (stmTrackerState *)malloc(sizeof(stmTrackerState));
    memset(state, 0, sizeof(stmTrackerState));
    stmTrackerState *result_state = (stmTrackerState *)malloc(sizeof(stmTrackerState));
    memset(result_state, 0, sizeof(stmTrackerState));

    float scale_x1 = VIDEO_PROCESS_WIDTH / PICTURE_PROCESS_WIDTH;
    float scale_y1 = VIDEO_PROCESS_HEIGHT / PICTURE_PROCESS_HEIGHT;
    // float x0 = 255.0f;
    // float y0 = 228.0f;
    // float x1 = 366.0f;
    // float y1 = 336.0f;
    float x0 = 40.0f;
    float y0 = 30.0f;
    float x1 = 100.0f;
    float y1 = 110.0f;

    state->cx = (x0 + x1) * 0.5f;
    state->cy = (y0 + y1) * 0.5f;
    state->w = x1 - x0;
    state->h = y1 - y0;
    state->scale = 1.0f;

    // state->cx = 670.0f + 0.5f * 145.0f;
    // state->cy = 234.0f + 0.5f * 113.0f;
    // state->w = 145.0f;
    // state->h = 113.0f;
    // state->scale = 1.0f;

    state->cx *= scale_x1;
    state->cy *= scale_y1;
    state->w *= scale_x1;
    state->h *= scale_y1;


    // kalman filter
    if (USE_KALMAN_FILTER) {
        kf_init(state->cx, state->cy, 0.0f, 0.0f);
    }

    ot_svp_img imgTemplate;  // for query model input
    ot_svp_img imgSearch;  // for memory model input

    ot_vb_blk vb_blk_template = createRgbFrame(&imgTemplate, TEMPLATE_FAKE_INPUT_SIZE, TEMPLATE_FAKE_INPUT_SIZE); 
    ot_vb_blk vb_blk_search = createRgbFrame(&imgSearch, SEARCH_FAKE_INPUT_SIZE, SEARCH_FAKE_INPUT_SIZE);  

    size_t imgTemplateSize = imgTemplate.width * imgTemplate.height * 3;  // bytes for uint8 input
    size_t imgSearchSize = imgSearch.width * imgSearch.height * 3;

    float context_amount = 0.5f;
    td_bool is_init = TD_FALSE;
    td_bool is_detected = TD_FALSE; 
    while (g_thdThread) {
        long time_start = getms();
        ret = ss_mpi_vpss_get_chn_frame(vpss_grp, vpss_chn, &frame_info, milli_sec);
        if (ret != TD_SUCCESS) {
            sample_print("grp(%d) ss_mpi_vpss_get_chn_frame errno %#x\n", vpss_grp,
                         ret);
            continue;
        }

        if (USE_KALMAN_FILTER && !is_init) {
            float ux = 0.0f;
            float uy = 0.0f;
            kf_correct(ux, uy, 0.1);
            float kf_pred_cx, kf_pred_cy; 
            kf_get_predicted_position(&kf_pred_cx, &kf_pred_cy);
            state->cx = kf_pred_cx;
            state->cy = kf_pred_cy;
        }

        // crop
        long t_s_crop = getms();
        int crop[4] = {0};  // xywh
        if (!is_init) {
            siamfcppTemplateGetCrop(state, context_amount, crop);
        } else {
            siamfcppSearchGetCrop(state, context_amount, crop);
        }
        ot_svp_img imgCrop;
        ot_vb_blk vb_blk_crop = createYuv420spFrame(&imgCrop, crop[2], crop[3]);
        if (vb_blk_crop == OT_VB_INVALID_HANDLE) {
            sample_print("createYuv420spFrame %d-%d failed\n", crop[2], crop[3]);
            goto create_yuvFrame_failed;
        }
        yuv420spFrameCrop(&imgCrop, &frame_info, crop[0], crop[1]);
        long t_e_crop = getms();
        sample_print("crop done, time: %ld ms\n", t_e_crop - t_s_crop);

        // color conversion(YUV420SP->RGB)
        long t_s_color = getms();
        ot_svp_img imgRGB;
        ot_vb_blk vb_blk_rgb = createRgbFrame(&imgRGB, imgCrop.width, imgCrop.height); 
        if (vb_blk_rgb == OT_VB_INVALID_HANDLE) {
            sample_print("createRgbFrame %d-%d failed\n", imgCrop.width, imgCrop.height);
            goto create_rgbFrame_failed;
        }
        yuv420spFrame2rgb(&imgCrop, &imgRGB);
        long t_e_color = getms();
        sample_print("color conversion done, time: %ld ms\n", t_e_color - t_s_color);

        // resize
        long t_s_resize = getms();
        if (!is_init) {
            rgbFrame2resize(&imgRGB, &imgTemplate);
        } else {
            rgbFrame2resize(&imgRGB, &imgSearch);
        }
        long t_e_resize = getms();
        sample_print("resize done, time: %ld ms\n", t_e_resize - t_s_resize);

        long t_s_inference = getms();
        if (!is_init) {
            template_execute((td_void*)imgTemplate.virt_addr[0], imgTemplateSize);
            is_init = TD_TRUE;
        } else {
            search_execute((td_void*)imgSearch.virt_addr[0], imgSearchSize, state, result_state);
        }
        long t_e_inference = getms();
        sample_print("model inference done, time: %ld ms\n", t_e_inference - t_s_inference);

        is_detected = result_state->score > SCORE_THRESHOLD;
        if (USE_KALMAN_FILTER) {
            if (is_detected) {
                float meas_std = 0.1f;
                kf_correct(result_state->cx, result_state->cy, meas_std);
            } else {
                kf_correct_without_measurement();
            }
            float kf_cor_cx, kf_cor_cy;
            kf_get_corrected_position(&kf_cor_cx, &kf_cor_cy);
            result_state->cx = kf_cor_cx;
            result_state->cy = kf_cor_cy;
        }

        if (is_detected) {
            state = result_state; 
        }

        sample_print("       state: cx: %.2f, cy: %.2f, w: %.2f, h: %.2f, score: %.2f\n", state->cx, state->cy, state->w, state->h, state->score);
        sample_print("result_state: cx: %.2f, cy: %.2f, w: %.2f, h: %.2f, score: %.2f\n", result_state->cx, result_state->cy, result_state->w, result_state->h, result_state->score);

create_rgbFrame_failed:
        ss_mpi_sys_munmap(imgRGB.virt_addr[0], imgRGB.width * imgRGB.height * 3);
        ss_mpi_vb_release_blk(vb_blk_rgb); 

create_yuvFrame_failed:
        ss_mpi_sys_munmap(imgCrop.virt_addr[0], imgCrop.width * imgCrop.height * 3 / 2);
        ss_mpi_vb_release_blk(vb_blk_crop); 
        
        pthread_mutex_lock(&algolock);
        stateInfo->cx = state->cx;
        stateInfo->cy = state->cy;
        stateInfo->w = state->w;
        stateInfo->h = state->h;
        stateInfo->score = state->score;
        pthread_mutex_unlock(&algolock);

        ret = ss_mpi_vpss_release_chn_frame(vpss_grp, vpss_chn, &frame_info);
        if (ret != TD_SUCCESS) {
            sample_print("grp(%d) ss_mpi_vpss_release_chn_frame errno %#x\n",
                         vpss_grp, ret);
        }
        long time_done = getms();
        sample_print("done, time: %ld ms\n", time_done - time_start);
    }
    siamfcpp_cleanup(); // unload model, etc

    if (state) {
        free(state);
    }

    ss_mpi_sys_munmap(imgTemplate.virt_addr[0], imgTemplateSize);
    ss_mpi_vb_release_blk(vb_blk_template); 
    ss_mpi_sys_munmap(imgSearch.virt_addr[0], imgSearchSize);
    ss_mpi_vb_release_blk(vb_blk_search); 
}


static int vgsdrawV2(ot_video_frame_info* pframe, stmTrackerState* state) {
    td_s32 ret;
    ot_vgs_handle h_handle = -1;
    ot_vgs_task_attr vgs_task_attr = { 0 };
    static ot_vgs_line stLines[4]; // 1 box = 4 lines
    int thick = 8;
    int color = 0x00FF00; // Green

    int xs = (int)(state->cx - state->w * 0.5f) & ~1; // align even
    int ys = (int)(state->cy - state->h * 0.5f) & ~1;
    int xe = (int)(state->cx + state->w * 0.5f) & ~1;
    int ye = (int)(state->cy + state->h * 0.5f) & ~1;

    // draw the 4 edges of the bounding box
    stLines[0] = (ot_vgs_line){.color=color, .thick=thick, .start_point={xs, ys}, .end_point={xe, ys}}; // top
    stLines[1] = (ot_vgs_line){.color=color, .thick=thick, .start_point={xs, ys}, .end_point={xs, ye}}; // left
    stLines[2] = (ot_vgs_line){.color=color, .thick=thick, .start_point={xe, ys}, .end_point={xe, ye}}; // right
    stLines[3] = (ot_vgs_line){.color=color, .thick=thick, .start_point={xs, ye}, .end_point={xe, ye}}; // bottom

    ret = ss_mpi_vgs_begin_job(&h_handle);
    if (ret != TD_SUCCESS) {
        return TD_FAILURE;
    }

    if (memcpy_s(&vgs_task_attr.img_in, sizeof(ot_video_frame_info), pframe,
        sizeof(ot_video_frame_info)) != EOK) {
        return TD_FAILURE;
    }

    if (memcpy_s(&vgs_task_attr.img_out, sizeof(ot_video_frame_info), pframe,
        sizeof(ot_video_frame_info)) != EOK) {
        return TD_FAILURE;
    }

    // Draw the box
    ret = ss_mpi_vgs_add_draw_line_task(h_handle, &vgs_task_attr, stLines, 4);
    if (ret != TD_SUCCESS) {
        ss_mpi_vgs_cancel_job(h_handle);
        printf("ss_mpi_vgs_add_draw_line_task ret:%08X\n", ret);
        return TD_FAILURE;
    }

    // Complete the VGS job
    ret = ss_mpi_vgs_end_job(h_handle);
    if (ret != TD_SUCCESS) {
        ss_mpi_vgs_cancel_job(h_handle);
        return TD_FAILURE;
    }

    return ret;
}

void *siamfcpp_draw_run(void *parg) {
    td_s32 ret;
    ot_video_frame_info frame_info = {0};
    const ot_vpss_grp vpss_grp = g_vpssGrp;
    const ot_vpss_chn vpss_chn = g_vpssChn;
    const td_s32 milli_sec = 200;
    stmTrackerState *state = (stmTrackerState *)malloc(sizeof(stmTrackerState));

    int policy;
    struct sched_param param;
    pthread_getschedparam(pthread_self(), &policy, &param);
    policy = SCHED_FIFO;
    param.sched_priority = sched_get_priority_max(policy);
    pthread_setschedparam(pthread_self(), policy, &param);

    while (g_thdThread) {
        ret = ss_mpi_vpss_get_chn_frame(vpss_grp, vpss_chn, &frame_info, milli_sec);
        if (ret != TD_SUCCESS) {
            continue;
        }
        pthread_mutex_lock(&algolock);
        memcpy(state, stateInfo, sizeof(stmTrackerState));
        pthread_mutex_unlock(&algolock);

        vgsdrawV2(&frame_info, state);

        ss_mpi_vo_send_frame(0, 0, &frame_info, milli_sec);
        ret = ss_mpi_vpss_release_chn_frame(vpss_grp, vpss_chn, &frame_info);
        if (ret != TD_SUCCESS) {
            sample_print("grp(%d) ss_mpi_vpss_release_chn_frame errno %#x\n",
                         vpss_grp, ret);
        }
    }

}