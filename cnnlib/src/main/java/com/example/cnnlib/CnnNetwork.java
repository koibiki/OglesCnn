package com.example.cnnlib;

import android.util.Log;

import com.example.cnnlib.eglenv.GLES31BackEnv;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.utils.DataUils;

import java.util.ArrayList;

import static android.os.Looper.getMainLooper;
import static com.example.cnnlib.utils.Constants.S_TEXTURE_SIZE;

/**
 * 网络最多只能保存7层数据
 */
public class CnnNetwork {

    private static final String TAG = "CnnNetwork";

    private ArrayList<Layer> mLayers;

    public CnnNetwork() {
        init();
        this.mLayers = new ArrayList<>();
    }

    private void init() {
        GLES31BackEnv gles31BackEnv = new GLES31BackEnv(S_TEXTURE_SIZE, S_TEXTURE_SIZE);
    }

    public void addLayer(Layer layer) throws Exception {
        int size = mLayers.size();
        if (size > 0) {
            Layer last = mLayers.get(size - 1);
        }
        mLayers.add(layer);
    }

    public void removeLayer(Layer layer) {
        mLayers.remove(layer);
    }

    public void run() {
        StringBuilder sb = new StringBuilder();
        long begin = System.currentTimeMillis();
        for (int i = 0; i < mLayers.size(); i++) {
            long begin1 = System.currentTimeMillis();
            mLayers.get(i).forwardProc();
            sb.append(i).append(":").append(System.currentTimeMillis() - begin1).append("; ");
        }
        Log.d(TAG, sb.toString());
        Log.w(TAG, "----- total spent:" + (System.currentTimeMillis() - begin));
        readOutput();
    }

    public void readOutput() {
        Layer layer = mLayers.get(mLayers.size() - 1);
        DataUils.readOutput(layer);
    }


}
