#include <stdio.h>

#include "sample_comm.h"
#include "sample_common_ive.h"
#include "sample_npu_process.h"

#include "stmTrack.h"
#include "stmTrack_process.h"
#include "createFrame.h"
#include "frameProcess.h"

#define MODEL_INPUT_SIZE 289 
#define MODEL_FAKE_INPUT_SIZE 304 

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

static void stmtrackGetCrop(stmTrackerState* state, float search_area_factor, int crop[4]) {
    // 4 times bigger (Todd Howard^_^)
    float search_size = search_area_factor * sqrtf(state->w * state->h);

    // ensure size is a multiple of 16
    search_size = (float)next_multiple_of_16((int)search_size);

    // update state scale
    state->scale = search_size / MODEL_INPUT_SIZE;

    // calculate crop rectangle
    crop[0] = (int)(state->cx - search_size * 0.5f);  // crop x
    crop[1] = (int)(state->cy - search_size * 0.5f);  // crop y
    crop[2] = (int)search_size;  // crop w
    crop[3] = (int)search_size;  // crop h

    // force crop x to be even since crop on YUV420SP
    crop[0] &= ~1;
}

void setMemoryMask(ot_svp_img* img, int crop[4], stmTrackerState *state) {
    // sanity check
    if (img->width != MODEL_INPUT_SIZE || img->height != MODEL_INPUT_SIZE || img->type != OT_SVP_IMG_TYPE_U8C1) {
        printf("Error: image size/type mismatch in setMemoryMask()\n");
        return;
    }

    // get crop position in original frame
    int crop_x = crop[0];
    int crop_y = crop[1];

    // compute target box position inside crop (original scale)
    float target_x_in_crop = state->cx - crop_x;
    float target_y_in_crop = state->cy - crop_y;

    // scale target box into 289*289
    float scale = state->scale; // this is search_size / 289, already provided
    float scaled_w = state->w / scale;
    float scaled_h = state->h / scale;
    float scaled_cx = target_x_in_crop / scale;
    float scaled_cy = target_y_in_crop / scale;

    // top-left and bottom-right corners in the 289*289 mask
    int x1 = (int)(scaled_cx - scaled_w * 0.5);
    int y1 = (int)(scaled_cy - scaled_h * 0.5);
    int x2 = (int)(scaled_cx + scaled_w * 0.5);
    int y2 = (int)(scaled_cy + scaled_h * 0.5);

    // clip to mask boundaries(just for safety)
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 >= MODEL_INPUT_SIZE) x2 = MODEL_INPUT_SIZE - 1;
    if (y2 >= MODEL_INPUT_SIZE) y2 = MODEL_INPUT_SIZE - 1;

    // clear whole mask to backgournd(0)
    memset((void*)img->virt_addr[0], 0, MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);

    // set target area to foreground(1)
    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            ((td_u8*)img->virt_addr[0])[y * MODEL_INPUT_SIZE + x] = 1;
        }
    }
}

void stmTrack_proc_init(int32_t vpss_grp, int32_t vpss_chn) {
    g_vpssGrp = vpss_grp;
    g_vpssChn = vpss_chn;
    g_thdThread = 1;
    stateInfo = (stmTrackerState*)malloc(sizeof(stmTrackerState));
}

void stmTrack_proc_uninit(void) {
    g_thdThread = 0;
}

void *stmTrack_proc_run(void *parg) {
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

    stmTrack_modleInit();

    stmTrackerState *state = (stmTrackerState *)malloc(sizeof(stmTrackerState));
    memset(state, 0, sizeof(stmTrackerState));

    float scale_x1 = VIDEO_PROCESS_WIDTH / PICTURE_PROCESS_WIDTH;
    float scale_y1 = VIDEO_PROCESS_HEIGHT / PICTURE_PROCESS_HEIGHT;

    state->cx = 670.0f + 0.5f * 145.0f;
    state->cy = 234.0f + 0.5f * 113.0f;
    state->w = 145.0f;
    state->h = 113.0f;
    state->scale = 1.0f;

    state->cx *= scale_x1;
    state->cy *= scale_y1;
    state->w *= scale_x1;
    state->h *= scale_y1;

    ot_svp_img imgQuery;  // for query model input
    ot_svp_img imgMemory;  // for memory model input
    ot_svp_img imgMask;  // for memory mask input

    ot_vb_blk vb_blk_query = createRgbFrame(&imgQuery, MODEL_FAKE_INPUT_SIZE, MODEL_FAKE_INPUT_SIZE); 
    ot_vb_blk vb_blk_memory = createRgbFrame(&imgMemory, MODEL_FAKE_INPUT_SIZE, MODEL_FAKE_INPUT_SIZE);  
    ot_vb_blk vb_blk_mask = createGrayFrame(&imgMask, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);  

    size_t imgQuerySize = imgQuery.width * imgQuery.height * 3;  // bytes for uint8 input
    size_t imgMemorySize = imgMemory.width * imgMemory.height * 3;
    size_t imgMaskSize = imgMask.width * imgMask.height;

    float search_area_factor = 4.0f;
    td_bool is_init = TD_FALSE;
    while (g_thdThread) {
        long time_start = getms();
        ret = ss_mpi_vpss_get_chn_frame(vpss_grp, vpss_chn, &frame_info, milli_sec);
        if (ret != TD_SUCCESS) {
            sample_print("grp(%d) ss_mpi_vpss_get_chn_frame errno %#x\n", vpss_grp,
                         ret);
            continue;
        }

        int crop[4] = {0};  // xywh
        stmtrackGetCrop(state, search_area_factor, crop);

        ot_svp_img imgCrop;
        ot_vb_blk vb_blk_crop = createYuv420spFrame(&imgCrop, crop[2], crop[3]);
        if (vb_blk_crop == OT_VB_INVALID_HANDLE) {
            sample_print("createYuv420spFrame %d-%d failed\n", crop[2], crop[3]);
            goto create_yuvFrame_failed;
        }
        yuv420spFrameCrop(&imgCrop, &frame_info, crop[0], crop[1]);

        ot_svp_img imgRGB;
        ot_vb_blk vb_blk_rgb = createRgbFrame(&imgRGB, imgCrop.width, imgCrop.height); 
        if (vb_blk_rgb == OT_VB_INVALID_HANDLE) {
            sample_print("createRgbFrame %d-%d failed\n", imgCrop.width, imgCrop.height);
            goto create_rgbFrame_failed;
        }
        yuv420spFrame2rgb(&imgCrop, &imgRGB);

        if (!is_init) {
            rgbFrame2resize(&imgRGB, &imgQuery);
            setMemoryMask(&imgMask, crop, state);
            is_init = TD_TRUE;
        } else {
            rgbFrame2resize(&imgRGB, &imgQuery);
        }

        stmTrack_execute((td_void*)imgQuery.virt_addr[0], imgQuerySize,
                         (td_void*)imgMemory.virt_addr[0], imgMemorySize, 
                         (td_void*)imgMask.virt_addr[0], imgMaskSize, 
                         state);

        sample_print("cx: %.2f, cy: %.2f, w: %.2f, h: %.2f, score: %.2f\n", state->cx, state->cy, state->w, state->h, state->score);

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

    if (state) {
        free(state);
    }

    ss_mpi_sys_munmap(imgQuery.virt_addr[0], imgQuery.width * imgQuery.height * 3);
    ss_mpi_vb_release_blk(vb_blk_query); 
    ss_mpi_sys_munmap(imgMemory.virt_addr[0], imgMemory.width * imgMemory.height * 3);
    ss_mpi_vb_release_blk(vb_blk_memory); 
    ss_mpi_sys_munmap(imgMask.virt_addr[0], imgMask.width * imgMask.height);
    ss_mpi_vb_release_blk(vb_blk_mask); 
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

void *stmTrack_draw_run(void *parg) {
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