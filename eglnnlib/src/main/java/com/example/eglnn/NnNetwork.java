package com.example.eglnn;

import android.content.Context;
import android.opengl.GLES31;
import android.util.Log;

import com.example.eglnn.eglenv.GLES31BackEnv;
import com.example.eglnn.layer.Layer;

import java.util.ArrayList;

import static com.example.eglnn.utils.Constants.S_TEXTURE_SIZE;

public class NnNetwork {

    private static final String TAG = "CnnNetwork";

    private Context mContext;
    private ArrayList<Layer> mLayers;

    private volatile boolean mIsInit = false;
    private GLES31BackEnv mGLes31BackEnv;

    public NnNetwork(Context context) {
        this.mContext = context;
        this.mLayers = new ArrayList<>();
        this.mGLes31BackEnv = new GLES31BackEnv(S_TEXTURE_SIZE, S_TEXTURE_SIZE);
    }

    public void addLayer(Layer layer) {
        mLayers.add(layer);
    }

    public void initialize() {
        if (!mIsInit) {
            mIsInit = true;
            for (Layer layer : mLayers) {
                layer.initialize();
                Log.w(TAG, layer.getName() + ":initialize");
            }
        }
    }

    public Result predict(float[][] input) {
        return runNet(input);
    }

    private Result runNet(float[][] input) {
        long begin = System.currentTimeMillis();
        float[] result = null;
        int count = 100;
        int filter = 10;
        for (int ii = 0; ii < count + filter; ii++) {
            long begin1 = System.currentTimeMillis();
            if (ii == filter) {
                begin = begin1;
            }
            int size = mLayers.size();
            for (int i = 0; i < size; i++) {
                mLayers.get(i).forwardProc(input, i + 1 == size);
                GLES31.glFlush();
            }
            result = actualReadOutput();
            Log.w(TAG, "spent:" + (System.currentTimeMillis() - begin1));
        }
        float aveTime = (System.currentTimeMillis() - begin) * 1.0f / count;
        Log.w(TAG, "ave spent:" + aveTime);
        return new Result(aveTime, result);
    }

    private float[] actualReadOutput() {
        Layer layer = mLayers.get(mLayers.size() - 1);
        return layer.readResult();
    }

    public static class Result {
        float aveTime;
        float[] result;

        Result(float aveTime, float[] result) {
            this.aveTime = aveTime;
            this.result = result;
        }

        public float getAveTime() {
            return aveTime;
        }

        public float[] getResult() {
            return result;
        }

    }

    public void destory() {
        mGLes31BackEnv.destroy();
    }

}
