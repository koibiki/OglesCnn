package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.utils.Constants;

public abstract class Layer {

    protected Context mContext;
    protected LayerParams mLayerParams;


    Layer(Context context, LayerParams layerParams) {
        this.mContext = context;
        this.mLayerParams = layerParams;
    }

    protected int mOutputTexID;
    protected int mAttachID;

    public LayerParams getLayerParams() {
        return mLayerParams;
    }

    public int getAttachID() {
        return mAttachID;
    }

    public int getOutputTexID() {
        return mOutputTexID;
    }

    public abstract int forwardProc(int inputTexID);

}
