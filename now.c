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

int src_x1 = x_crop >= 0 ? x_crop : 0;
int src_y1 = y_crop >= 0 ? y_crop : 0;
int src_x2 = x_crop + w_crop <= srcf->video_frame.width ? x_crop + w_crop : srcf->video_frame.width;
int src_y2 = y_crop + h_crop <= srcf->video_frame.height ? y_crop + h_crop : srcf->video_frame.height;

int crop_w = src_x2 - src_x1;  
int crop_h = src_y2 - src_y1;  

int dst_x = x_crop < 0 ? -x_crop : 0;
int dst_y = y_crop < 0 ? -y_crop : 0;

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

dst_data.phys_addr = dstf->phys_addr[1];
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