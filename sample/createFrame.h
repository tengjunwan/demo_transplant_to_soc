#ifndef _CREATEFRAME_H__
#define _CREATEFRAME_H__

#include <stdio.h>
#include "sample_comm.h"
#include "sample_common_svp.h"

ot_vb_blk createYuv420spFrame(ot_svp_img* img, td_s32 w, td_s32 h);
ot_vb_blk createRgbFrame(ot_svp_img* img, td_s32 w, td_s32 h);
ot_vb_blk createGrayFrame(ot_svp_img* img, td_s32 w, td_s32 h);

#endif