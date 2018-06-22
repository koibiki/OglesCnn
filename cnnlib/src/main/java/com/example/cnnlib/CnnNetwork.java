package com.example.cnnlib;

import android.util.Log;

import com.example.cnnlib.eglenv.GLES31BackEnv;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.utils.DataUtils;

import java.util.ArrayList;

import static com.example.cnnlib.utils.Constants.S_TEXTURE_SIZE;

/**
 * 网络只有8个缓存能够保存数据
 */
public class CnnNetwork {

    private static final String TAG = "CnnNetwork";

    private ArrayList<Layer> mLayers;

    private boolean mIsInit;
    private GLES31BackEnv mGLes31BackEnv;

    public CnnNetwork() {
        init();
        this.mLayers = new ArrayList<>();
    }

    private void init() {
        mGLes31BackEnv = new GLES31BackEnv(S_TEXTURE_SIZE, S_TEXTURE_SIZE);
    }

    public void addLayer(Layer layer) {
        mLayers.add(layer);
    }

    public void initialize() {
        mGLes31BackEnv.post(this::actualInitialize);
    }

    private void actualInitialize() {
        if (!mIsInit) {
            mIsInit = true;
            for (Layer layer : mLayers) {
                layer.initialize();
            }
        }
    }

    public void removeLayer(Layer layer) {
        mLayers.remove(layer);
    }

    public void run() {
        mGLes31BackEnv.post(this::actualRun);
    }

    private void actualRun() {
        StringBuilder sb = new StringBuilder();
        long begin = System.currentTimeMillis();
        for (int i = 0; i < mLayers.size(); i++) {
            long begin1 = System.currentTimeMillis();
            mLayers.get(i).forwardProc(false);
            sb.append(i).append(":").append(System.currentTimeMillis() - begin1).append("; ");
        }
        Log.d(TAG, sb.toString());
        Log.w(TAG, "----- total spent:" + (System.currentTimeMillis() - begin));

        actualReadOutput();
    }

    public void readOutput() {
        mGLes31BackEnv.post(this::actualReadOutput);
    }

    private void actualReadOutput() {
        long begin = System.currentTimeMillis();
        Layer layer = mLayers.get(mLayers.size() - 1);
        DataUtils.readOutput(layer);
        Log.w(TAG, "read spent:" + (System.currentTimeMillis() - begin));
    }


}
