package com.example.cnnlib.utils;

import java.util.ArrayList;
import java.util.List;

public class AttachIDManager {

    private int mMaxBuffer;
    private static AttachIDManager mAttachIDManager;
    // 计算数据的 attachId 为 0,1,2
    private int mCurrentDataId = 2;

    // int[] x1,y1, x2,y2 存储方式为起点x1 和 其对角点x2
    List<int[]> mConvKennelRestoredIndex = new ArrayList<>();

    // conv kennel 的 attachId 为 3 所有conv层均存储在1个纹理上
    private int mConvId = 3;

    // 全连接的 kennel 每层单独存在1个纹理上
    private int mCurrentFullConnKennelId = 4;

    private AttachIDManager() {

    }

    public static AttachIDManager getInstance() {
        if (mAttachIDManager == null) {
            mAttachIDManager = new AttachIDManager();
        }
        return mAttachIDManager;
    }

    public void initialize(int maxBuffer) {
        this.mMaxBuffer = maxBuffer;
    }

    public int getConvAttachId() {
        return mConvId;
    }

    public int getFullConnKennelAttachId() {
        if (mCurrentDataId < mMaxBuffer) {
            try {
                throw new Exception("全连接层数超过最大范围");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return mCurrentFullConnKennelId++;
    }

    public int getDataAttachID() {
        mCurrentDataId++;
        return mCurrentDataId % 3;
    }

    public int[] getLastConvKennelIndex() {
        int size = mConvKennelRestoredIndex.size();
        if (size == 0) {
            return new int[]{0, 0, 0, 0};
        } else {
            return mConvKennelRestoredIndex.get(size - 1);
        }
    }

    public void saveConvKennelIndex(int[] indexes) {
        mConvKennelRestoredIndex.add(indexes);
    }

}
