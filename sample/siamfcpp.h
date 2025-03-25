#ifndef _SIAMFCPP_H__
#define _SIAMFCPP_H__

#include <stdio.h>

#define TEST_VIDEO_WIDTH 				1920.0f
#define TEST_VIDEO_HEIGHT 			1080.0f
#define VIDEO_PROCESS_WIDTH 		640.0f
#define VIDEO_PROCESS_HEIGHT 		640.0f
// #define PICTURE_PROCESS_WIDTH 	960.0f
// #define PICTURE_PROCESS_HEIGHT 	540.0f
#define PICTURE_PROCESS_WIDTH 	800.0f
#define PICTURE_PROCESS_HEIGHT 	600.0f

void siamfcpp_proc_init(int32_t vpss_grp, int32_t vpss_chn);
void siamfcpp_proc_uninit(void);
void *siamfcpp_proc_run(void *parg);
void *siamfcpp_draw_run(void *parg);
#endif