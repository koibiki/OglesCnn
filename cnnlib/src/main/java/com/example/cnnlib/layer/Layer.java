package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.utils.Constants;

public abstract class Layer {

    protected Context mContext;
    protected LayerParams mLayerParams;

    public Layer(Context context, LayerParams layerParams) {
        this.mContext = context;
        this.mLayerParams = layerParams;
    }

    protected int mOutputTexID;

    public LayerParams getLayerParams(){
        return mLayerParams;
    }

    public int getOutputTexID() {
        return mOutputTexID;
    }

    public abstract int forwardProc(int InputTexID, int attachID);


}
