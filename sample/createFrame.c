#include "createFrame.h"

ot_vb_blk createYuv420spFrame(ot_svp_img* img, td_s32 w, td_s32 h) {
    td_s32 vbsize = w * h * 3 / 2;

    ot_vb_blk vb_blk = ss_mpi_vb_get_blk(OT_VB_INVALID_POOL_ID, vbsize, TD_NULL);
    if (vb_blk == OT_VB_INVALID_HANDLE) {
        sample_print("Error: Failed to allcoate VB block for YUV420SP!\n");
        return OT_VB_INVALID_HANDLE;
    }

    img->phys_addr[0] = ss_mpi_vb_handle_to_phys_addr(vb_blk);
    if (img->phys_addr[0] == 0) {
        sample_print("Error: Failed to get physical address for YUV420SP!\n");
        ss_mpi_vb_release_blk(vb_blk);
        return OT_VB_INVALID_HANDLE;
    }
    img->phys_addr[1] = img->phys_addr[0] + w * h;
    img->virt_addr[0] = (td_u64)(td_u8*)ss_mpi_sys_mmap(
        img->phys_addr[0], vbsize);
    if (img->virt_addr[0] == 0) {
        sample_print("Error: Failed to get virtual address for YUV420SP!\n");
        ss_mpi_vb_release_blk(vb_blk);
        return OT_VB_INVALID_HANDLE;
    }
    img->virt_addr[1] = img->virt_addr[0] + w * h;

    // zero init
    memset((void*)img->virt_addr[0], 0, vbsize);  

    // set metadata
    img->stride[0] = w;
    img->stride[1] = w;
    img->width = w;
    img->height = h;
    img->type = OT_SVP_IMG_TYPE_YUV420SP;

    // sample_print("create YUV frame: w=%d, h=%d, stride[0]=%d, stride[1]=%d\n", img->width, img->height, img->stride[0], img->stride[1]);

    return vb_blk; 
}

ot_vb_blk createRgbFrame(ot_svp_img* img, td_s32 w, td_s32 h) {
    td_s32 vbsize = w * h * 3;

    ot_vb_blk vb_blk = ss_mpi_vb_get_blk(OT_VB_INVALID_POOL_ID, vbsize, TD_NULL);
    if (vb_blk == OT_VB_INVALID_HANDLE) {
        printf("Error: Failed to allcoate VB block! for RGB frame\n");
        return OT_VB_INVALID_HANDLE;
    }

    img->phys_addr[0] = ss_mpi_vb_handle_to_phys_addr(vb_blk); // B plane
    img->phys_addr[1] = img->phys_addr[0] + w * h; // G plane
    img->phys_addr[2] = img->phys_addr[1] + w * h; // R plane
    if (img->phys_addr[0] == 0) {
        sample_print("Error: Failed to get physical address! for RGB frame\n");
        ss_mpi_vb_release_blk(vb_blk);
        return OT_VB_INVALID_HANDLE;
    }

    img->virt_addr[0] = (td_u64)(td_u8*)ss_mpi_sys_mmap(img->phys_addr[0], vbsize);
    img->virt_addr[1] = img->virt_addr[0] + w * h; // G plane
    img->virt_addr[2] = img->virt_addr[1] + w * h; // R plane
    if (img->virt_addr[0] == 0) {
        sample_print("Error: Failed to get virtual address! for RGB frame\n");
        ss_mpi_vb_release_blk(vb_blk);
        return OT_VB_INVALID_HANDLE;
    }

    memset((void*)img->virt_addr[0], 0, vbsize);

    img->stride[0] = w;
    img->stride[1] = w;
    img->stride[2] = w;
    img->width = w;
    img->height = h;
    img->type = OT_SVP_IMG_TYPE_U8C3_PLANAR;

    // sample_print("create RGB frame: w=%d, h=%d, stride[0]=%d, stride[1]=%d, stride[2]=%d\n", 
    //              img->width, img->height, img->stride[0], img->stride[1], img->stride[2]);

    return vb_blk;
}

ot_vb_blk createGrayFrame(ot_svp_img* img, td_s32 w, td_s32 h) {
    td_s32 vbsize = w * h;

    ot_vb_blk vb_blk = ss_mpi_vb_get_blk(OT_VB_INVALID_POOL_ID, vbsize, TD_NULL);
    if (vb_blk == OT_VB_INVALID_HANDLE) {
        sample_print("Error: Failed to allcoate VB block! for Gray frame\n");
        return OT_VB_INVALID_HANDLE;
    }

    img->phys_addr[0] = ss_mpi_vb_handle_to_phys_addr(vb_blk); // single plane
    if (img->phys_addr[0] == 0) {
        sample_print("Error: Failed to get physical address! for Gray frame\n");
        ss_mpi_vb_release_blk(vb_blk);
        return OT_VB_INVALID_HANDLE;
    }

    img->virt_addr[0] = (td_u64)(td_u8*)ss_mpi_sys_mmap(img->phys_addr[0], vbsize);
    if (img->virt_addr[0] == 0) {
        sample_print("Error: Failed to get virtual address! for Gray frame\n");
        ss_mpi_vb_release_blk(vb_blk);
        return OT_VB_INVALID_HANDLE;
    }

    memset((void*)img->virt_addr[0], 0, vbsize);

    img->stride[0] = w;
    img->width = w;
    img->height = h;
    img->type = OT_SVP_IMG_TYPE_U8C1;

    // sample_print("create Gray frame: w=%d, h=%d, stride[0]=%d\n", 
    //              img->width, img->height, img->stride[0]);

    return vb_blk;
}