package com.example.cnnlib;

import android.util.Log;

import com.example.cnnlib.eglenv.GLES31BackEnv;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.utils.DataUils;

import java.util.ArrayList;

import static android.os.Looper.getMainLooper;
import static com.example.cnnlib.utils.Constants.S_TEXTURE_SIZE;

public class CnnNetwork {

    private static final String TAG = "CnnNetwork";

    private ArrayList<Layer> mLayers;

    public CnnNetwork() {
        init();
        this.mLayers = new ArrayList<>();
    }

    private void init() {
        GLES31BackEnv gles31BackEnv = new GLES31BackEnv(S_TEXTURE_SIZE, S_TEXTURE_SIZE);
        gles31BackEnv.setThreadOwner(getMainLooper().getThread().getName());
    }

    public void addLayer(Layer layer) throws Exception {
        int size = mLayers.size();
        if (size > 0) {
            Layer last = mLayers.get(size - 1);
            int[] in = layer.getLayerParams().inputShape;
            int[] out = last.getLayerParams().outputShape;
            for (int i = 0; i < in.length; i++) {
                if (in[i] != out[i]) {
                    throw new Exception("层间维度不一致");
                }
            }
        }
        mLayers.add(layer);
    }

    public void removeLayer(Layer layer) {
        mLayers.remove(layer);
    }

    public void run() {
        int currentTexID = 0;
        long begin = System.currentTimeMillis();
        for (int i = 0; i < mLayers.size(); i++) {
            long begin1 = System.currentTimeMillis();
            currentTexID = mLayers.get(i).forwardProc(currentTexID);
            Log.d(TAG, " spent:" + (System.currentTimeMillis() - begin1));
        }
        Log.w(TAG, "----- total spent:" + (System.currentTimeMillis() - begin));
    }

    public void readOutput() {
        Layer layer = mLayers.get(mLayers.size() - 1);
        DataUils.readOutput(layer);
    }


}
