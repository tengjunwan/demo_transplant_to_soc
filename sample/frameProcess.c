#include "frameProcess.h"

td_s32 yuv420spFrame2rgb(ot_svp_img *srcYUV, ot_svp_img *dstRGB) {
    td_s32 ret;
    ot_ive_handle handle;
    ot_ive_csc_ctrl csc_ctrl;

    csc_ctrl.mode = OT_IVE_CSC_MODE_PIC_BT709_YUV_TO_RGB;

    ret = ss_mpi_ive_csc(&handle, (const ot_svp_src_img*)srcYUV, (const ot_svp_dst_img*)dstRGB, 
                          (const ot_ive_csc_ctrl*)&csc_ctrl, TD_TRUE);  // require resolution as the multiple of 16
    if (ret != TD_SUCCESS) {
        printf("Error: YUV420SP to RGB conversion failed!\n");
        return ret;
    }

    td_bool is_finish = TD_FALSE;
    td_bool is_block = TD_TRUE;
    ret = ss_mpi_ive_query(handle, &is_finish, is_block);
    while (ret == OT_ERR_IVE_QUERY_TIMEOUT) {
        usleep(100);
        ret  = ss_mpi_ive_query(handle, &is_finish, is_block);
    }

    return TD_SUCCESS;
}

td_s32 rgbFrame2resize(ot_svp_img *srcRGB, ot_svp_img *dstRGB) {
    td_s32 ret;
    ot_ive_handle handle;
    ot_ive_resize_ctrl resize_ctrl;

    // sample_print("srcRGB: w=%d, h=%d, stride[0]=%d, stride[1]=%d, stride[2]=%d\n",
    //        srcRGB->width, srcRGB->height, srcRGB->stride[0], srcRGB->stride[1], srcRGB->stride[2]);
    // sample_print("dstRGB: w=%d, h=%d, stride[0]=%d, stride[1]=%d, stride[2]=%d\n",
    //        dstRGB->width, dstRGB->height, dstRGB->stride[0], dstRGB->stride[1], dstRGB->stride[2]);
    // sample_print("srcRGB type: %d, dstRGB type: %d\n", srcRGB->type, dstRGB->type);

    // image arrays
    ot_svp_img src_array[1] = {*srcRGB};
    ot_svp_img dst_array[1] = {*dstRGB};

    // set up resize control parameters
    resize_ctrl.mode = OT_IVE_RESIZE_MODE_LINEAR; // bilinear interpolation
    resize_ctrl.num = 1; // processing a single image

    // allocate extra memory required
    td_u32 U8C1_NUM = 0; // since we're dealing with U8C3_PLANAR
    td_u32 mem_size = 25 * U8C1_NUM + 49 * (resize_ctrl.num - U8C1_NUM);

    memset(&resize_ctrl.mem, 0, sizeof(resize_ctrl.mem)); //  zero out
    td_s32 ret_mem = ss_mpi_sys_mmz_alloc(&resize_ctrl.mem.phys_addr, (td_void**)&resize_ctrl.mem.virt_addr,
                                          "resize_mem", TD_NULL, mem_size);
    resize_ctrl.mem.size = mem_size; // need to be set manually!
    if (ret_mem != TD_SUCCESS) {
        sample_print("Errror: failed to allocate memory for resize_ctrl.mem!\n");
        return TD_FAILURE;
    }

    // performing resizing
    ret = ss_mpi_ive_resize(&handle, src_array, dst_array, &resize_ctrl, TD_TRUE);
    if (ret != TD_SUCCESS) {
        sample_print("Error: RGB resize failed!\n");
        sample_print("error code: 0x%08X\n", ret);
        ss_mpi_sys_mmz_free(resize_ctrl.mem.phys_addr, resize_ctrl.mem.virt_addr);
        return ret;
    }

    // Query to check if the resizing is finished
    td_bool is_finish = TD_FALSE;
    td_bool is_block = TD_TRUE;
    ret = ss_mpi_ive_query(handle, &is_finish, is_block);
    while (ret == OT_ERR_IVE_QUERY_TIMEOUT) {
        usleep(100);
        ret = ss_mpi_ive_query(handle, &is_finish, is_block);
    }
    // free the allocated memory
    ss_mpi_sys_mmz_free(resize_ctrl.mem.phys_addr, resize_ctrl.mem.virt_addr);

    return TD_SUCCESS;
}

td_s32 yuv420spFrameCrop(ot_svp_dst_img* dstf, ot_video_frame_info* srcf, 
                         td_s32 x_crop, td_s32 y_crop) {
    td_s32 ret = OT_ERR_IVE_NULL_PTR;
    ot_ive_handle handle;
    ot_svp_dst_data dst_data;
    ot_svp_src_data src_data;
    td_bool is_finish = TD_FALSE;
    td_bool is_block = TD_TRUE;
    ot_ive_dma_ctrl ctrl = { OT_IVE_DMA_MODE_DIRECT_COPY, 0, 0, 0, 0}; 

    int w_crop = dstf->width;
    int h_crop = dstf->height;
    
    // determine cropping range inside source frame
    int src_x1 = x_crop >= 0 ? x_crop : 0;
    int src_y1 = y_crop >= 0 ? y_crop : 0;
    int src_x2 = x_crop + w_crop <= srcf->video_frame.width ? x_crop + w_crop : srcf->video_frame.width;
    int src_y2 = y_crop + h_crop <= srcf->video_frame.height ? y_crop + h_crop : srcf->video_frame.height;

    int crop_w = src_x2 - src_x1;  // actual crop width
    int crop_h = src_y2 - src_y1;  // actual crop height

    // determine dst offset if crop starts out-of-bounds (black padding)
    int dst_x = x_crop < 0 ? -x_crop : 0;
    int dst_y = y_crop < 0 ? -y_crop : 0;

    // copy Y plane
    src_data.phys_addr = srcf->video_frame.phys_addr[0] + src_y1 * srcf->video_frame.stride[0] + src_x1;
    src_data.width = crop_w;
    src_data.height = crop_h;
    src_data.stride = srcf->video_frame.stride[0];

    dst_data.phys_addr = dstf->phys_addr[0] + dst_y * dstf->stride[0] + dst_x;
    dst_data.width = crop_w;
    dst_data.height = crop_h;
    dst_data.stride = dstf->stride[0];
    
    ret = ss_mpi_ive_dma(&handle, &src_data, &dst_data, &ctrl, TD_TRUE);
    if (ret != TD_SUCCESS) return -1;

    ret = ss_mpi_ive_query(handle, &is_finish, is_block);
    while (ret == OT_ERR_IVE_QUERY_TIMEOUT) {
        usleep(100);
        ret = ss_mpi_ive_query(handle, &is_finish, is_block);
    }

    // copy UV plane
    src_data.phys_addr = srcf->video_frame.phys_addr[1] + (src_y1 / 2) * srcf->video_frame.stride[1] + src_x1;
    src_data.width = crop_w;
    src_data.height = crop_h / 2;
    src_data.stride = srcf->video_frame.stride[1];

    dst_data.phys_addr = dstf->phys_addr[1] + (dst_y / 2) * dstf->stride[1] + dst_x;
    dst_data.width = crop_w;
    dst_data.height = crop_h / 2;
    dst_data.stride = dstf->stride[1];

    ret = ss_mpi_ive_dma(&handle, &src_data, &dst_data, &ctrl, TD_TRUE);
    if (ret != TD_SUCCESS) return -1;

    ret = ss_mpi_ive_query(handle, &is_finish, is_block);
    while (ret == OT_ERR_IVE_QUERY_TIMEOUT) {
        usleep(100);
        ret = ss_mpi_ive_query(handle, &is_finish, is_block);
    }

    return TD_SUCCESS;
}


td_s32 videoFrame2rgb(ot_video_frame_info* srcf, ot_svp_img* dstRGB) {
    td_s32 ret;
    ot_ive_handle handle;
    ot_ive_csc_ctrl csc_ctrl;

    // step 1: wrap ot_video_frame_info (YUV420SP) into ot_svp_img
    ot_svp_img srcYUV;
    memset(&srcYUV, 0, sizeof(ot_svp_img));

    srcYUV.type = OT_SVP_IMG_TYPE_YUV420SP;
    srcYUV.width = srcf->video_frame.width;
    srcYUV.height = srcf->video_frame.height;
    srcYUV.stride[0] = srcf->video_frame.stride[0];
    srcYUV.stride[1] = srcf->video_frame.stride[1];


    srcYUV.phys_addr[0] = srcf->video_frame.phys_addr[0];  // Y
    srcYUV.phys_addr[1] = srcf->video_frame.phys_addr[1];  // UV

    srcYUV.virt_addr[0] = srcf->video_frame.virt_addr[0];
    srcYUV.virt_addr[1] = srcf->video_frame.virt_addr[1];  

    // step 2: setup color space conversion (YUV -> RGB)
    csc_ctrl.mode = OT_IVE_CSC_MODE_PIC_BT709_YUV_TO_RGB;

    // step 3: perform the conversion
    ret = ss_mpi_ive_csc(&handle, (const ot_svp_src_img*)&srcYUV,
                         (const ot_svp_dst_img*)dstRGB,
                         &csc_ctrl, TD_TRUE);
    
    if (ret != TD_SUCCESS) {
        sample_print("Error: YUV420SP to RGB conversion failed! ret=0x%08X\n", ret);
        return ret;
    }
}


td_s32 rgbFrameCrop(ot_svp_dst_img* dstf, ot_svp_dst_img* srcf, td_s32 x_crop, td_s32 y_crop) {
    td_s32 ret = OT_ERR_IVE_NULL_PTR;
    ot_ive_handle handle;
    ot_svp_dst_data dst_data;
    ot_svp_src_data src_data;
    td_bool is_finish = TD_FALSE;
    td_bool is_block = TD_TRUE;
    ot_ive_dma_ctrl ctrl = { OT_IVE_DMA_MODE_DIRECT_COPY, 0, 0, 0, 0}; 

    int w_crop = dstf->width;
    int h_crop = dstf->height;

    // determine cropping range inside source frame
    int src_x1 = x_crop >= 0 ? x_crop : 0;
    int src_y1 = y_crop >= 0 ? y_crop : 0;
    int src_x2 = x_crop + w_crop <= srcf->width ? x_crop + w_crop : srcf->width;
    int src_y2 = y_crop + h_crop <= srcf->height ? y_crop + h_crop : srcf->height;

    int crop_w = src_x2 - src_x1;  // actual crop width
    int crop_h = src_y2 - src_y1;  // actual crop height

    // determine dst offset if crop starts out-of-bounds (black padding)
    int dst_x = x_crop < 0 ? -x_crop : 0;
    int dst_y = y_crop < 0 ? -y_crop : 0;

    for (int c = 0; c < 3; c++) {
        src_data.phys_addr = srcf->phys_addr[c] + src_y1 * srcf->stride[c] + src_x1;
        src_data.width = crop_w;
        src_data.height = crop_h;
        src_data.stride = srcf->stride[c];

        dst_data.phys_addr = dstf->phys_addr[c] + dst_y * dstf->stride[c] + dst_x;
        dst_data.width = crop_w;
        dst_data.height = crop_h;
        dst_data.stride = dstf->stride[c];

        ret = ss_mpi_ive_dma(&handle, &src_data, &dst_data, &ctrl, TD_TRUE);
        if (ret != TD_SUCCESS) return -1;

        ret = ss_mpi_ive_query(handle, &is_finish, is_block);
        while (ret == OT_ERR_IVE_QUERY_TIMEOUT) {
            usleep(100);
            ret = ss_mpi_ive_query(handle, &is_finish, is_block);
        }   
    }
    return TD_SUCCESS;
}


void clear_svp_imgRGB(ot_svp_img* img) {
    for (int i = 0; i < 3; i++) {
        if (img->virt_addr[i] != TD_NULL) {
            memset(img->virt_addr[i], 0, img->stride[i] * img->height);
        }
    }
}