package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;

public class SoftMaxLayer extends Layer {

    private static final String TAG = "SoftMaxLayer";

    SoftMaxLayer(Context context, LayerParams layerParams) {
        super(context, layerParams);
    }

    @Override
    public int forwardProc(int inTex) {
        return 0;
    }

    @Override
    public void readOutput() {

    }

}
