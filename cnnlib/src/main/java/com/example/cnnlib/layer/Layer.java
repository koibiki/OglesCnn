package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;

public abstract class Layer {

    protected Context mContext;
    protected LayerParams mLayerParams;


    Layer(Context context, LayerParams layerParams) {
        this.mContext = context;
        this.mLayerParams = layerParams;
    }

    protected int mOutTex;
    protected int mAttachID;

    public LayerParams getLayerParams() {
        return mLayerParams;
    }

    public int getAttachID() {
        return mAttachID;
    }

    public int getOutTex() {
        return mOutTex;
    }

    public abstract int forwardProc(int inTex);

    public abstract void readOutput();

}
