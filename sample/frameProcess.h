#ifndef _FRAMEPROCESS_H__
#define _FRAMEPROCESS_H__

#include <stdio.h>
#include "sample_comm.h"
#include "sample_common_ive.h"

td_s32 yuv420spFrame2rgb(ot_svp_img *srcYUV, ot_svp_img *dstRGB);
td_s32 rgbFrame2resize(ot_svp_img *srcRGB, ot_svp_img *dstRGB);
td_s32 yuv420spFrameCrop(ot_svp_dst_img* dstf, ot_video_frame_info* srcf, 
                         td_s32 x_crop, td_s32 y_crop);

#endif