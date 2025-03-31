#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "sample_comm.h"
#include "stmTrack.h"
#include "siamfcpp.h"

#define VO_LAYER 0
#define VO_CHN 0
#define VDEC_CHN 0
#define VDEC_CHN_NUM 1
#define VPSS_GRP 0
#define VPSS_CHN 0
#define VENC_CHN 0

static volatile sig_atomic_t g_sig_flag = 0;
static pthread_t g_vdec_thread;
static pthread_t nnn_pid;
static pthread_t draw_pid;

static sample_vdec_attr g_vdec_cfg = {
    .type = OT_PT_H265,
    .mode = OT_VDEC_SEND_MODE_FRAME,
    .width = FHD_WIDTH,
    .height = FHD_HEIGHT,
    .sample_vdec_video.dec_mode = OT_VIDEO_DEC_MODE_IP,
    .sample_vdec_video.bit_width = OT_DATA_BIT_WIDTH_8,
    .sample_vdec_video.ref_frame_num = 2+1, //不加1的话 自定义的mp4文件会花屏
    .display_frame_num = 2,               
    .frame_buf_cnt = 5,                   
};

static vdec_thread_param g_vdec_thread_param = {
    .chn_id = 0,
    .type = OT_PT_H265,
    .stream_mode = OT_VDEC_SEND_MODE_FRAME,
    .interval_time = 1000, 
    .pts_init = 0,
    .pts_increase = 0,
    .e_thread_ctrl = THREAD_CTRL_START,
    .circle_send = TD_TRUE,
    .milli_sec = 0,
    .min_buf_size = (FHD_WIDTH * FHD_HEIGHT * 3) >> 1, 
    .c_file_path = "res",
    // .c_file_name = "polo.h265",
    // .c_file_name = "mario_with_pipes.h265",
    .c_file_name = "balloon.h265",
    .fps = 30, 
};

static sample_vo_cfg g_vo_cfg = {
    .vo_dev = SAMPLE_VO_DEV_UHD,
    .vo_intf_type = OT_VO_INTF_HDMI,
    .intf_sync = OT_VO_OUT_1080P30,
    .bg_color = COLOR_RGB_BLACK,
    .pix_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    .disp_rect = {0, 0, 1920, 1080},
    .image_size = {1920, 1080},
    .vo_part_mode = OT_VO_PARTITION_MODE_MULTI,
    .dis_buf_len = 3, /* 3: def buf len for single */
    .dst_dynamic_range = OT_DYNAMIC_RANGE_SDR8,
    .vo_mode = VO_MODE_2MUX,
    .compress_mode = OT_COMPRESS_MODE_NONE,
};

static td_void sample_get_char(td_void) {
    if (g_sig_flag == 1) {
        return;
    }
    sample_pause();
}

static td_s32 sample_start_vdec(const ot_size *size) {
    td_s32 ret;
    ot_pic_buf_attr buf_attr = { 0 };
    ot_vb_cfg vb_cfg;
    td_u32 chn_num = VDEC_CHN_NUM;

    buf_attr.align = 0;
    buf_attr.bit_width = OT_DATA_BIT_WIDTH_8;
    buf_attr.compress_mode = OT_COMPRESS_MODE_NONE;
    buf_attr.width = size->width;
    buf_attr.height = size->height;
    buf_attr.pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;

    (td_void)memset_s(&vb_cfg, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg));
    vb_cfg.max_pool_cnt = 1;                               
    vb_cfg.common_pool[0].blk_cnt = 20 * chn_num; 
    vb_cfg.common_pool[0].blk_size = ot_common_get_pic_buf_size(&buf_attr);
    ret = sample_comm_sys_init(&vb_cfg);
    if (ret != TD_SUCCESS) {
        sample_print("init sys fail for %#x!\n", ret);
        sample_comm_sys_exit();
        return ret;
    }

    ret = sample_comm_vdec_init_vb_pool(chn_num, &g_vdec_cfg, OT_VDEC_MAX_CHN_NUM);
    if (ret != TD_SUCCESS) {
        sample_print("vdec init vb_pool fail\n");
        sample_comm_sys_exit();
        return ret;
    }
    ret = sample_comm_vdec_start(chn_num, &g_vdec_cfg, OT_VDEC_MAX_CHN_NUM);
    if (ret != TD_SUCCESS) {
        sample_print("vdec start fail\n");
        sample_comm_vdec_exit_vb_pool();
        sample_comm_sys_exit();
    }

    return ret;
}

static td_s32 sample_stop_vdec(td_void) {
    td_s32 ret;
    td_u32 chn_num = VDEC_CHN_NUM;

    ret = sample_comm_vdec_stop(chn_num);
    if (ret != TD_SUCCESS) {
        printf("vdec stop fail\n");
        return TD_FAILURE;
    }
    sample_comm_vdec_exit_vb_pool();
    return TD_SUCCESS;
}

static td_s32 sample_start_vpss(td_s32 vpss_grp, td_s32 vpss_chn, const ot_size *size) {
    td_s32 ret;
    ot_vpss_grp_attr grp_attr;
    ot_vpss_chn_attr chn_attr[OT_VPSS_MAX_PHYS_CHN_NUM] = {0};
    td_bool chn_enable[OT_VPSS_MAX_PHYS_CHN_NUM] = {0};

    sample_comm_vpss_get_default_grp_attr(&grp_attr);
    grp_attr.max_width = size->width;
    grp_attr.max_height = size->height;
    sample_comm_vpss_get_default_chn_attr(&chn_attr[vpss_chn]);
    chn_attr[vpss_chn].width = VIDEO_PROCESS_WIDTH;
    chn_attr[vpss_chn].height = VIDEO_PROCESS_HEIGHT;
    chn_enable[vpss_chn] = TD_TRUE;
    chn_attr[vpss_chn].compress_mode = OT_COMPRESS_MODE_NONE;
    chn_attr[vpss_chn].depth = 1;

    sample_comm_vpss_get_default_chn_attr(&chn_attr[vpss_chn+1]);
    chn_attr[vpss_chn+1].width = VIDEO_PROCESS_WIDTH;
    chn_attr[vpss_chn+1].height = VIDEO_PROCESS_HEIGHT;
    chn_enable[vpss_chn+1] = TD_TRUE;
    chn_attr[vpss_chn+1].compress_mode = OT_COMPRESS_MODE_NONE;
    chn_attr[vpss_chn+1].depth = 1;

    ret = sample_common_vpss_start(vpss_grp, chn_enable, &grp_attr, chn_attr,
        OT_VPSS_MAX_PHYS_CHN_NUM);
    if (ret != TD_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_stop_vpss(td_s32 vpss_grp) {
    td_s32 ret;
    td_bool chn_enable[OT_VPSS_MAX_PHYS_CHN_NUM] = {0};

    ret = sample_common_vpss_stop(vpss_grp, chn_enable, OT_VPSS_MAX_PHYS_CHN_NUM);
    if (ret != TD_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_void sample_start_send_stream(td_void) {
    td_u32 chn_num = VDEC_CHN_NUM;

    sample_comm_vdec_start_send_stream(chn_num, &g_vdec_thread_param, &g_vdec_thread, OT_VDEC_MAX_CHN_NUM,
        2 * OT_VDEC_MAX_CHN_NUM); /* 2:thread num */
}

static td_void sample_stop_send_stream(td_void) {
    td_u32 chn_num = VDEC_CHN_NUM;

    sample_comm_vdec_stop_send_stream(chn_num, &g_vdec_thread_param, &g_vdec_thread, OT_VDEC_MAX_CHN_NUM,
        2 * OT_VDEC_MAX_CHN_NUM); /* 2:thread num */
    sleep(1);
}

static td_s32 sample_start_all(td_void) {
    td_s32 ret;
    ot_vdec_chn vdec_chn = VDEC_CHN;
    ot_vpss_grp vpss_grp = VPSS_GRP; 
    ot_vpss_chn vpss_chn = VPSS_CHN;
    ot_vo_layer vo_layer = VO_LAYER;
    ot_vo_chn vo_chn = VO_CHN;
    ot_size size;
    // size.width = TEST_VIDEO_WIDTH;
    // size.height = TEST_VIDEO_HEIGHT;
    size.width = 8000;
    size.height = 6000;

    ret = sample_start_vdec(&size);
    if (ret != TD_SUCCESS) {
        sample_print("start vdec failed with 0x%x!\n", ret);
        return ret;
    }

    ret = sample_start_vpss(vpss_grp, vpss_chn, &size);
    if (ret != TD_SUCCESS) {
        sample_print("start vpss failed with 0x%x!\n", ret);
        return TD_FAILURE;
    }

    ret = sample_comm_vdec_bind_vpss(vdec_chn, vpss_grp);
    if (ret != TD_SUCCESS) {
        sample_print("vi_bind_multi_vpss 0x%x!\n", ret);
        return TD_FAILURE;
    }

    sample_comm_vo_get_def_config(&g_vo_cfg);
    ret = sample_comm_vo_start_vo(&g_vo_cfg);
    if (ret != TD_SUCCESS) {
        sample_print("start vo failed with 0x%x!\n", ret);
        return TD_FAILURE;
    }
    // ret = sample_comm_vpss_bind_vo(vpss_grp, vpss_chn, vo_layer, vo_chn);
    // if (ret != TD_SUCCESS) {
    //     sample_print("vpss bind vo failed with 0x%x!\n", ret);
    //     return TD_FAILURE;
    // }

    sample_start_send_stream();

    // stmTrack_proc_init(vpss_grp, vpss_chn);
    // pthread_create(&nnn_pid, 0, stmTrack_proc_run, NULL);
    // pthread_create(&draw_pid, 0, stmTrack_draw_run, NULL);

    siamfcpp_proc_init(vpss_grp, vpss_chn);
    pthread_create(&nnn_pid, 0, siamfcpp_proc_run, NULL);
    pthread_create(&draw_pid, 0, siamfcpp_draw_run, NULL);
}

static td_s32 sample_stop_all(td_void) {
    td_s32 ret;
    ot_vdec_chn vdec_chn = VDEC_CHN;
    ot_vpss_grp vpss_grp = VPSS_GRP; 
    ot_vpss_chn vpss_chn = VPSS_CHN;
    ot_vo_layer vo_layer = VO_LAYER;
    ot_vo_chn vo_chn = VO_CHN;

    ret = sample_comm_vpss_un_bind_vo(vpss_grp, vdec_chn, vo_layer, vo_chn);
    if (ret != TD_SUCCESS) {
        sample_print("sample_comm_vpss_un_bind_vo failed with 0x%x!\n", ret);
        return ret;
    }
    ret = sample_comm_vo_stop_vo(&g_vo_cfg);
    if (ret != TD_SUCCESS) {
        sample_print("sample_comm_vo_stop_vo failed with 0x%x!\n", ret);
        return ret;
    }
    ret = sample_comm_vdec_un_bind_vpss(vdec_chn, vpss_grp);
    if (ret != TD_SUCCESS) {
        sample_print("sample_comm_vdec_un_bind_vpss failed with 0x%x!\n", ret);
        return ret;
    }
    sample_stop_send_stream();
    ret = sample_stop_vpss(vpss_grp);
    if (ret != TD_SUCCESS) {
        sample_print("sample_comm_vdec_un_bind_vpss failed with 0x%x!\n", ret);
        return ret;
    }
    ret = sample_stop_vdec();
    if (ret != TD_SUCCESS) {
        sample_print("sample_region_stop_vdec failed with 0x%x!\n", ret);
        return ret;
    }

    return TD_SUCCESS;
}

static td_void sample_vdec_handle_sig(td_s32 signo) {
    if (signo == SIGINT || signo == SIGTERM) {
        g_sig_flag = 1;
    }
}

static td_void sample_register_sig_handler(td_void(*sig_handle)(td_s32)) {
    struct sigaction sa;

    (td_void)memset_s(&sa, sizeof(struct sigaction), 0, sizeof(struct sigaction));
    sa.sa_handler = sig_handle;
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, TD_NULL);
    sigaction(SIGTERM, &sa, TD_NULL);
}

td_s32 main(td_s32 argc, td_char* argv[]) {
    td_s32 ret;
    sample_register_sig_handler(sample_vdec_handle_sig);
    ret = sample_start_all();
    if ((ret == TD_SUCCESS) && (g_sig_flag == 0)) {
        printf("\033[0;32mprogram exit normally!\033[0;39m\n");
    }
    else {
        printf("\033[0;31mprogram exit abnormally!\033[0;39m\n");
    }

    sample_get_char();

    // stmTrack_proc_uninit();
    siamfcpp_proc_uninit();
    sample_stop_all();

    exit(ret);
}


