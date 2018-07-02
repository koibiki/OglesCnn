package com.example.cnnlib;

import android.opengl.GLES31;
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

    private volatile boolean mIsInit = false;
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

    public void predict(float[][][] input) {
        mGLes31BackEnv.post(() -> actualPredict(input));
    }


    private void actualPredict(float[][][] input) {
        runNet(input);
        actualReadOutput();
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


    private void runNet(float[][][] input) {
        long begin = System.currentTimeMillis();
        int count = 1000;
        for (int ii = 0; ii < count; ii++) {

            long begin1 = System.currentTimeMillis();
            for (int i = 0; i < mLayers.size(); i++) {
                mLayers.get(i).forwardProc(input);
                GLES31.glFlush();
            }
            actualReadOutput();
            Log.w(TAG, "spent:" + (System.currentTimeMillis() - begin1));
        }
        Log.w(TAG, "ave spent:" + (System.currentTimeMillis() - begin) / count);
    }

    public void readOutput() {
        mGLes31BackEnv.post(this::actualReadOutput);
    }

    private void actualReadOutput() {
        Layer layer = mLayers.get(mLayers.size() - 1);
        DataUtils.readOutput(layer);
    }


}
