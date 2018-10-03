package com.example.eglnn.layer;

import android.content.Context;

public class GlobalAvePooling extends Pooling {
    public GlobalAvePooling(Context context, String name, Layer preLayer) {
        super(context, name, preLayer, preLayer.getOutputShape()[0], preLayer.getOutputShape()[1], PaddingType.VALID, preLayer.getOutputShape()[0], preLayer.getOutputShape()[1], PoolingType.AVE);
    }
}
